"""Shared TRPO CLI options and unambiguous experiment names."""
def add_trpo_options(parser):
    parser.add_argument("--trpo-critic-regime", choices=["ours", "gfn_pg"], default="ours")
    parser.add_argument("--trpo-delta", type=float, default=0.01)
    parser.add_argument("--trpo-cg-iters", type=int, default=10)
    parser.add_argument("--trpo-cg-damping", type=float, default=0.0)
    parser.add_argument("--trpo-line-search-iters", type=int, default=10)
    parser.add_argument("--trpo-line-search-shrink", type=float, default=0.5)


def trpo_name(args):
    return (f"trpo_{args.trpo_critic_regime}_delta{args.trpo_delta:g}"
            f"_cg{args.trpo_cg_iters}_damp{args.trpo_cg_damping:g}"
            f"_ls{args.trpo_line_search_iters}_shrink{args.trpo_line_search_shrink:g}")


def trpo_cli(args):
    return (f"--trpo-critic-regime {args.trpo_critic_regime} --trpo-delta {args.trpo_delta} "
            f"--trpo-cg-iters {args.trpo_cg_iters} --trpo-cg-damping {args.trpo_cg_damping} "
            f"--trpo-line-search-iters {args.trpo_line_search_iters} "
            f"--trpo-line-search-shrink {args.trpo_line_search_shrink} ")
