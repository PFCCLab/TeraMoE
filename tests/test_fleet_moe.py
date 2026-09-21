import os
import sys

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)

import teramoe
import teramoe.fused_a2a as teramoe_fused_a2a

# use develop paddlefleet
FLEET_PATH = os.path.realpath("../../erniebot_test_speed/third_party/PaddleFleet/src")
sys.path.insert(0, FLEET_PATH)
os.environ["FLEET_MOE_EP_BARRIER_ASYNC"] = "1"
import paddlefleet
assert paddlefleet.__path__[0] == os.path.join(FLEET_PATH, "paddlefleet"), (
    f"Unexpected paddlefleet path: {paddlefleet.__path__[0]}")
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe import MoELayer, MoESublayers
import paddlefleet.transformer.moe.fused_a2a as fleet_fused_a2a
from paddlefleet.transformer.transformer_config import TransformerConfig

SEQLEN = 16384
NUM_COMM_SMS = 48
NUM_CALC_SMS = 100


def initialize_fleet():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "sharding_degree": world_size,
        "pp_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "dp_degree": 1,
        "ep_degree": world_size,
        "mp_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    paddlefleet.training.initialize.initialize_fleet(strategy=strategy)
    model_parallel_cuda_manual_seed(rank)
    group = ProcessGroupCollection.use_mpu_process_groups()
    return group


class TeraMoELayer(MoELayer):
    def fusion_moe_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        combine_overlap_handle: dict,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        hidden_states = self._project_to_latent(hidden_states)

        hidden_states = teramoe.forward_autograd(
            hidden_states,
            topk_weights,
            topk_indices,
            self.grouped_gemm_experts,
            self.num_experts,
            self.moe_group,
            combine_overlap_handle,
            num_calc_sms=NUM_CALC_SMS,
        )

        if self.use_latent_moe:
            if self.latent_norm is not None:
                hidden_states = self.latent_norm(hidden_states)
            hidden_states = self.fc2_latent_proj(hidden_states)

        return hidden_states


def run_layer(moe_layer, hidden_states, out_grad, profile=None):
    # warmup
    hidden_states = hidden_states.detach()
    hidden_states.stop_gradient = False
    # 真实情况下传入的 hs 是经过 norm 的, 不是 leaf 节点, 这里用 clone 模拟 norm
    with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
        out, _ = moe_layer(hidden_states.clone())
    out.backward(out_grad)

    w1 = moe_layer.grouped_gemm_experts.weight1
    w2 = moe_layer.grouped_gemm_experts.weight2
    w1.grad.zero_()
    w2.grad.zero_()
    # 让 test 充分 overlap, 暴露出 overlap 可能存在的问题
    dist.all_reduce(paddle.empty([1]))

    # test
    hidden_states = hidden_states.detach()
    hidden_states.stop_gradient = False
    with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
        out, _ = moe_layer(hidden_states.clone())
    out.backward(out_grad)

    if profile is None:
        return out, hidden_states.grad, w1.grad, w2.grad

    # profile
    paddle.base.core.nvprof_nvtx_push(profile)

    for i in range(10):
        hidden_states = hidden_states.detach()
        hidden_states.stop_gradient = False
        hidden_states_t = hidden_states.clone()

        dist.all_reduce(paddle.empty([1]))
        paddle.base.core.nvprof_nvtx_push("forward")
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            out, _ = moe_layer(hidden_states_t)
        paddle.base.core.nvprof_nvtx_pop()

        dist.all_reduce(paddle.empty([1]))
        paddle.base.core.nvprof_nvtx_push("backward")
        out.backward(out_grad)
        paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_nvtx_pop()


def main():
    group = initialize_fleet()
    fleet_fused_a2a.configure_buffer(NUM_COMM_SMS)
    teramoe.configure_buffer(NUM_COMM_SMS)

    config = TransformerConfig(
        hidden_size=4096,
        moe_intermediate_size=2048,  # for both routed and shared experts
        gated_linear_unit=True,
        moe_latent_size=2048,
        latent_moe_use_norm=True,
        n_routed_experts=512,
        n_shared_experts=1,
        num_experts_per_tok=10,
        topk_method="noaux_tc",
        moe_token_dispatcher_type="deepep",
        # fp8="e4m3",
        fp8_wgrad=False,
        moe_topk_fusion=True,
        routing_map_fusion=True,
        sigmoid_gate_fusion=True,
        moe_expert_fusion=True,
        moe_shared_expert_overlap=True,
        # hidden_act="situ",
        # situ_glu_fusion=True,
        # situ_glu_plain_fusion=True,
    )

    mlp_spec = MLPSublayersSpec(
        up_gate_proj=ColumnParallelLinear,
        down_proj=RowParallelLinear,
    )

    hidden_states = paddle.randn([1, SEQLEN, config.hidden_size], dtype="bfloat16")
    out_grad = (paddle.randn(hidden_states.shape) * 0.02).cast("bfloat16")

    ############################### FLEET LAYER ################################

    moe_layer = MoELayer(config, MoESublayers(mlp_spec), group)
    moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
    state_dict = moe_layer.state_dict()

    print("-" * 80)
    print("moe_layer:", moe_layer)
    for name, param in moe_layer.named_parameters():
        print("param:", name, list(param.shape), param.dtype.name.lower())
    print("-" * 80)

    fleet_out = run_layer(moe_layer, hidden_states, out_grad)

    fleet_fused_a2a._buffer = None
    dist.barrier()

    ############################## TERAMOE LAYER ###############################

    moe_layer = TeraMoELayer(config, MoESublayers(mlp_spec), group)
    moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
    moe_layer.set_state_dict(state_dict)

    teramoe_out = run_layer(moe_layer, hidden_states, out_grad)

    teramoe_fused_a2a._buffer = None
    dist.barrier()

    names = ["out", "hs_grad", "w1_grad", "w2_grad"]
    ok = True
    for name, ref, tgt in zip(names, fleet_out, teramoe_out):
        diff = (ref.float() - tgt.float()).abs()
        max_, avg_ = float(diff.max()), float(diff.mean())
        ok = ok and max_ == 0
        print(f"{name}: max={max_} avg={avg_}")

    ################################# PROFILE ##################################

    paddle.base.core.nvprof_start()

    # fleet
    moe_layer = MoELayer(config, MoESublayers(mlp_spec), group)
    moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
    state_dict = moe_layer.state_dict()

    run_layer(moe_layer, hidden_states, out_grad, profile="fleet")

    fleet_fused_a2a._buffer = None
    dist.barrier()

    # teramoe
    moe_layer = TeraMoELayer(config, MoESublayers(mlp_spec), group)
    moe_layer = paddle.amp.decorate(moe_layer, level="O2", dtype="bfloat16")
    moe_layer.set_state_dict(state_dict)

    run_layer(moe_layer, hidden_states, out_grad, profile="teramoe")

    dist.barrier()
    paddle.base.core.nvprof_stop()

    print("PASSED" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
