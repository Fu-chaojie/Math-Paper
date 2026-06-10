"""问题1：机场交通中心拥堵瓶颈识别。

运行：python q1_bottleneck_analysis.py
输出：Mathcode/outputs/q1/ 下的 CSV、PNG 与结果摘要。
"""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODES = ("taxi", "ride_hailing", "private_car", "bus")
OUTPUT = Path(__file__).resolve().parent / "outputs" / "q1"

# 图表统一使用中文，便于直接写入论文。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CN_CITY = {"Hongqiao": "上海虹桥机场", "Xiaoshan": "杭州萧山机场"}
CN_MODE = {
    "taxi": "出租车",
    "ride_hailing": "网约车",
    "private_car": "私家车",
    "bus": "机场巴士",
}
CN_ROLE = {
    "root_bottleneck": "根因瓶颈",
    "secondary_bottleneck": "次级瓶颈",
    "surface_congestion": "表面拥堵",
    "non_bottleneck": "非瓶颈",
}
CN_TYPE = {"road": "道路", "supply": "运力供给", "gate": "入口服务", "curb": "上客服务"}
LABEL_OFFSET = {
    "bus_gate": (5, 12),
    "bus_curb": (5, -16),
    "private_car_gate": (5, -16),
    "private_car_curb": (5, 12),
    "taxi_gate": (-8, -18),
    "taxi_curb": (5, 12),
    "ride_hailing_gate": (-8, 12),
    "ride_hailing_curb": (5, -18),
}


def cn_resource(name):
    """将程序内部资源名转换为论文图表所用中文名称。"""
    fixed = {
        "entry_road": "入口道路",
        "exit_road": "离场道路",
        "exit_storage": "离场道路缓冲空间",
    }
    if name in fixed:
        return fixed[name]
    for mode, label in CN_MODE.items():
        if name == mode:
            return f"{label}供给"
        if name == f"{mode}_gate":
            return f"{label}入口"
        if name == f"{mode}_curb":
            return f"{label}上客区"
        if name == f"{mode}_supply":
            return f"{label}供给"
    return name


# 全部参数均集中在此；后续获得真实调查数据时可直接替换。
CITIES = {
    "Hongqiao": {
        # 上海虹桥国际机场（SHA）：2024年旅客吞吐量47,944,067人次，
        # 起降架次275,288架次；以下参数校准为4小时高峰陆侧到达情景。
        "airport_code": "SHA",
        "annual_passengers_2024": 47_944_067,
        "annual_movements_2024": 275_288,
        "seed": 2026,
        "horizon": 240,
        "demand": {"base": 38, "peaks": [(70, 18, 62), (172, 26, 74)]},
        "share": dict(zip(MODES, (0.22, 0.38, 0.24, 0.16))),
        "load": dict(zip(MODES, (1.55, 1.65, 2.20, 32.0))),
        "supply_factor": dict(zip(MODES, (1.08, 1.10, 1.00, 1.04))),
        "supply_lag": dict(zip(MODES, (7, 6, 0, 10))),
        "entry_road": 36.0,
        "exit_road": 34.0,
        "exit_storage": 175.0,
        "gate": dict(zip(MODES, (14.5, 19.0, 12.0, 2.2))),
        "curb": dict(zip(MODES, (14.0, 18.5, 11.2, 1.9))),
        "stage": dict(zip(MODES, (120.0, 95.0, 45.0, 16.0))),
    },
    "Xiaoshan": {
        # 杭州萧山国际机场（HGH）：2024年旅客吞吐量48,053,915人次，
        # 起降架次320,269架次；与虹桥年客流接近，但交通中心结构不同。
        "airport_code": "HGH",
        "annual_passengers_2024": 48_053_915,
        "annual_movements_2024": 320_269,
        "seed": 2036,
        "horizon": 240,
        "demand": {"base": 36, "peaks": [(78, 22, 56), (184, 20, 72)]},
        "share": dict(zip(MODES, (0.24, 0.36, 0.22, 0.18))),
        "load": dict(zip(MODES, (1.55, 1.62, 2.15, 30.0))),
        "supply_factor": dict(zip(MODES, (0.98, 1.02, 0.98, 1.03))),
        "supply_lag": dict(zip(MODES, (12, 9, 0, 14))),
        "entry_road": 31.0,
        "exit_road": 30.0,
        "exit_storage": 160.0,
        "gate": dict(zip(MODES, (13.0, 15.5, 11.0, 2.1))),
        "curb": dict(zip(MODES, (12.8, 15.2, 10.5, 2.0))),
        "stage": dict(zip(MODES, (95.0, 85.0, 40.0, 16.0))),
    },
}


def validate_configs():
    """在仿真前检查参数完整性，防止校准数据录入错误。"""
    mode_fields = ("share", "load", "supply_factor", "supply_lag", "gate", "curb", "stage")
    for city, cfg in CITIES.items():
        assert abs(sum(cfg["share"].values()) - 1) < 1e-9, f"{city}: 方式分担率之和不为1"
        assert cfg["horizon"] > 0 and cfg["entry_road"] > 0 and cfg["exit_road"] > 0
        for field in mode_fields:
            assert set(cfg[field]) == set(MODES), f"{city}: {field} 的交通方式不完整"
            assert all(v >= 0 for v in cfg[field].values()), f"{city}: {field} 存在负值"


def export_calibration_parameters():
    """导出可直接用于论文附表的两机场校准参数。"""
    rows = []
    for city, cfg in CITIES.items():
        common = {
            "机场": CN_CITY[city],
            "代码": cfg["airport_code"],
            "2024旅客吞吐量": cfg["annual_passengers_2024"],
            "2024起降架次": cfg["annual_movements_2024"],
            "仿真时长_分钟": cfg["horizon"],
            "入口道路能力_辆每分钟": cfg["entry_road"],
            "离场道路能力_辆每分钟": cfg["exit_road"],
            "离场道路缓冲容量_辆": cfg["exit_storage"],
        }
        for m in MODES:
            rows.append(
                {
                    **common,
                    "交通方式": CN_MODE[m],
                    "方式分担率": cfg["share"][m],
                    "平均载客量": cfg["load"][m],
                    "供给响应系数": cfg["supply_factor"][m],
                    "供给延迟_分钟": cfg["supply_lag"][m],
                    "入口服务能力_辆每分钟": cfg["gate"][m],
                    "上客服务能力_辆每分钟": cfg["curb"][m],
                    "缓冲区容量_辆": cfg["stage"][m],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "calibration_parameters.csv", index=False, encoding="utf-8-sig")


def peak_curve(t, base, peaks):
    """以多个高斯峰近似航班集中到达所产生的客流波动。"""
    curve = np.full_like(t, base, dtype=float)
    for center, width, height in peaks:
        curve += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return curve


def generate_inputs(city, seed=None):
    """生成同一随机种子下可复现的旅客需求与车辆供给。"""
    cfg = CITIES[city]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    t = np.arange(cfg["horizon"])
    total_rate = peak_curve(t, **cfg["demand"])

    demand, supply = {}, {}
    for m in MODES:
        person_rate = total_rate * cfg["share"][m]
        demand[m] = rng.poisson(person_rate)

        # 车辆供给响应需求但存在调入延迟，城市差异由校准后的供给系数体现。
        lag = cfg["supply_lag"][m]
        base_vehicle_rate = person_rate / cfg["load"][m]
        vehicle_rate = np.empty_like(base_vehicle_rate)
        vehicle_rate[:lag] = base_vehicle_rate[0]
        vehicle_rate[lag:] = base_vehicle_rate[:-lag] if lag else base_vehicle_rate
        vehicle_rate *= cfg["supply_factor"][m]
        supply[m] = rng.poisson(np.maximum(vehicle_rate, 0.05))

    return pd.DataFrame(demand), pd.DataFrame(supply)


def proportional_limit(request, capacity):
    """容量不足时按请求比例分配，保持各交通方式之间的基本公平。"""
    total = sum(request.values())
    scale = min(1.0, capacity / total) if total > 0 else 1.0
    return {m: request[m] * scale for m in MODES}


def simulate(city, demand, supply, perturb=None):
    """有限容量双边排队网络：旅客队列、车辆队列与出口回溢相互耦合。"""
    cfg = deepcopy(CITIES[city])
    supply = supply.astype(float).copy()

    # 扰动某项资源10%，用于识别其对系统总延误的因果影响。
    if perturb:
        kind, name = perturb
        if kind == "supply":
            supply[name] *= 1.10
        elif name in ("entry_road", "exit_road", "exit_storage"):
            cfg[name] *= 1.10
        else:
            cfg[kind][name] *= 1.10

    passenger = {m: 0.0 for m in MODES}
    external = {m: 0.0 for m in MODES}
    stage = {m: 0.0 for m in MODES}
    exit_occupancy = 0.0
    logs = []

    for t in range(cfg["horizon"]):
        for m in MODES:
            passenger[m] += demand.at[t, m]
            external[m] += supply.at[t, m]

        # 下游道路先排出车辆；若出口存储空间不足，将反向阻塞上客区。
        exit_occupancy -= min(exit_occupancy, cfg["exit_road"])

        gate_raw = {
            m: min(external[m], max(cfg["stage"][m] - stage[m], 0.0)) for m in MODES
        }
        gate_request = {m: min(gate_raw[m], cfg["gate"][m]) for m in MODES}
        admitted = proportional_limit(gate_request, cfg["entry_road"])
        for m in MODES:
            external[m] -= admitted[m]
            stage[m] += admitted[m]

        curb_raw = {m: min(stage[m], passenger[m] / cfg["load"][m]) for m in MODES}
        curb_request = {m: min(curb_raw[m], cfg["curb"][m]) for m in MODES}
        exit_space = max(cfg["exit_storage"] - exit_occupancy, 0.0)
        served_vehicle = proportional_limit(curb_request, exit_space)

        served_person = {}
        for m in MODES:
            served_person[m] = min(passenger[m], served_vehicle[m] * cfg["load"][m])
            stage[m] -= served_vehicle[m]
            passenger[m] -= served_person[m]
        exit_occupancy += sum(served_vehicle.values())

        entry_block = sum(gate_request.values()) - sum(admitted.values())
        exit_block = sum(curb_request.values()) - sum(served_vehicle.values())
        record = {
            "time": t,
            "exit_occupancy": exit_occupancy,
            "entry_load": sum(gate_request.values()) / cfg["entry_road"],
            "exit_load": sum(curb_request.values()) / cfg["exit_road"],
            "entry_block": entry_block,
            "exit_block": exit_block,
        }
        for m in MODES:
            record.update(
                {
                    f"p_{m}": passenger[m],
                    f"external_{m}": external[m],
                    f"stage_{m}": stage[m],
                    f"gate_load_{m}": gate_raw[m] / cfg["gate"][m],
                    f"curb_load_{m}": curb_raw[m] / cfg["curb"][m],
                    f"gate_block_{m}": max(gate_raw[m] - gate_request[m], 0.0),
                    f"curb_block_{m}": max(curb_raw[m] - curb_request[m], 0.0),
                    f"served_person_{m}": served_person[m],
                }
            )
        logs.append(record)

    df = pd.DataFrame(logs)
    passenger_area = df[[f"p_{m}" for m in MODES]].sum(axis=1).sum()
    driver_area = df[
        [f"external_{m}" for m in MODES] + [f"stage_{m}" for m in MODES]
    ].sum(axis=1).sum()
    road_area = df["exit_occupancy"].sum()
    block_area = df[["entry_block", "exit_block"]].sum().sum()

    # 统一广义延误成本，用于容量扰动弹性比较。
    total_cost = passenger_area + 1.2 * driver_area + 2.0 * road_area + 4.0 * block_area
    return df, total_cost


def node_diagnostics(city, df):
    """计算负荷、排队和回溢，用于识别拥堵表象。"""
    cfg = CITIES[city]
    rows = [
        {
            "node": "entry_road",
            "mean_load": df["entry_load"].mean(),
            "p95_load": df["entry_load"].quantile(0.95),
            "max_queue": df[[f"external_{m}" for m in MODES]].sum(axis=1).max(),
            "spill_probability": (df["entry_block"] > 1e-6).mean(),
        },
        {
            "node": "exit_road",
            "mean_load": df["exit_load"].mean(),
            "p95_load": df["exit_load"].quantile(0.95),
            "max_queue": df["exit_occupancy"].max(),
            "spill_probability": (df["exit_block"] > 1e-6).mean(),
        },
    ]
    for m in MODES:
        rows.extend(
            [
                {
                    "node": f"{m}_gate",
                    "mean_load": df[f"gate_load_{m}"].mean(),
                    "p95_load": df[f"gate_load_{m}"].quantile(0.95),
                    "max_queue": df[f"external_{m}"].max(),
                    "spill_probability": (df[f"gate_block_{m}"] > 1e-6).mean(),
                },
                {
                    "node": f"{m}_curb",
                    "mean_load": df[f"curb_load_{m}"].mean(),
                    "p95_load": df[f"curb_load_{m}"].quantile(0.95),
                    "max_queue": df[f"p_{m}"].max(),
                    "spill_probability": (df[f"curb_block_{m}"] > 1e-6).mean(),
                },
            ]
        )
    result = pd.DataFrame(rows)
    result["storage_ratio"] = np.where(
        result["node"].eq("exit_road"),
        result["max_queue"] / cfg["exit_storage"],
        np.nan,
    )
    return result


def system_summary(city, demand, supply, df, cost):
    """基于Little定律，用队列面积估计系统平均等待时间。"""
    served = sum(df[f"served_person_{m}"].sum() for m in MODES)
    p_area = df[[f"p_{m}" for m in MODES]].sum().sum()
    v_area = df[
        [f"external_{m}" for m in MODES] + [f"stage_{m}" for m in MODES]
    ].sum().sum()
    return {
        "city": city,
        "generalized_cost": cost,
        "passenger_demand": demand.sum().sum(),
        "vehicle_supply": supply.sum().sum(),
        "served_passengers": served,
        "service_rate": served / demand.sum().sum(),
        "avg_passenger_wait_min": p_area / max(served, 1),
        "avg_vehicle_wait_min": v_area / max(supply.sum().sum(), 1),
        "max_passenger_queue": df[[f"p_{m}" for m in MODES]].sum(axis=1).max(),
        "max_vehicle_queue": df[
            [f"external_{m}" for m in MODES] + [f"stage_{m}" for m in MODES]
        ].sum(axis=1).max(),
        "max_exit_occupancy": df["exit_occupancy"].max(),
    }


def sensitivity(city, demand, supply, base_cost):
    """分别提高节点容量或车辆供给10%，计算成本下降率与瓶颈弹性。"""
    resources = [
        ("road", "entry_road"),
        ("road", "exit_road"),
        ("road", "exit_storage"),
        *[("gate", m) for m in MODES],
        *[("curb", m) for m in MODES],
        ("supply", "taxi"),
        ("supply", "ride_hailing"),
        ("supply", "bus"),
    ]
    rows = []
    for kind, name in resources:
        _, new_cost = simulate(city, demand, supply, (kind, name))
        reduction = (base_cost - new_cost) / base_cost
        rows.append(
            {
                "resource": f"{name}_{kind}" if kind not in ("road", "supply") else name,
                "type": kind,
                "cost_reduction_pct": 100 * reduction,
                "bottleneck_elasticity": reduction / 0.10,
            }
        )
    return pd.DataFrame(rows).sort_values("bottleneck_elasticity", ascending=False)


def stability_analysis(city, repeats=30):
    """多随机种子重复实验，检验瓶颈排序是否稳定。"""
    rows = []
    base_seed = CITIES[city]["seed"]
    for i in range(repeats):
        demand, supply = generate_inputs(city, base_seed + i)
        _, base_cost = simulate(city, demand, supply)
        ranking = sensitivity(city, demand, supply, base_cost)
        top_resource = ranking.iloc[0]["resource"]
        for _, row in ranking.iterrows():
            rows.append(
                {
                    "repeat": i + 1,
                    "resource": row["resource"],
                    "bottleneck_elasticity": row["bottleneck_elasticity"],
                    "is_top1": row["resource"] == top_resource,
                }
            )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("resource")
        .agg(
            mean_elasticity=("bottleneck_elasticity", "mean"),
            std_elasticity=("bottleneck_elasticity", "std"),
            positive_probability=("bottleneck_elasticity", lambda x: (x > 0).mean()),
            top1_probability=("is_top1", "mean"),
        )
        .reset_index()
        .sort_values("mean_elasticity", ascending=False)
    )
    return raw, summary


def joint_diagnosis(nodes, sensitivity_df):
    """联合表面拥堵与因果弹性，避免将回溢产生的长队误判为根因。"""
    diagnosis = nodes.copy()
    diagnosis["resource_type"] = "physical"
    diagnosis["resource"] = diagnosis["node"].str.replace(
        r"_(gate|curb)$", lambda x: "_" + x.group(1), regex=True
    )
    elasticity = sensitivity_df.set_index("resource")["bottleneck_elasticity"]
    diagnosis["causal_elasticity"] = diagnosis["resource"].map(elasticity).fillna(0.0)

    def scale(series):
        span = series.max() - series.min()
        return (series - series.min()) / span if span > 0 else series * 0

    diagnosis["surface_score"] = (
        0.4 * scale(diagnosis["p95_load"].clip(upper=4))
        + 0.3 * scale(diagnosis["max_queue"])
        + 0.3 * diagnosis["spill_probability"]
    )
    diagnosis["role"] = np.select(
        [
            diagnosis["causal_elasticity"] >= 0.15,
            diagnosis["causal_elasticity"] >= 0.02,
            (diagnosis["surface_score"] >= 0.45)
            & (diagnosis["causal_elasticity"] <= 0),
        ],
        ["root_bottleneck", "secondary_bottleneck", "surface_congestion"],
        default="non_bottleneck",
    )

    # 运力供给不是实体节点，但可能是造成旅客拥堵的根本资源瓶颈。
    for m in ("taxi", "ride_hailing", "bus"):
        elasticity_m = elasticity.get(m, 0.0)
        if elasticity_m <= 0.02:
            continue
        surface_m = diagnosis.loc[
            diagnosis["node"].eq(f"{m}_curb"), "surface_score"
        ].iloc[0]
        diagnosis.loc[len(diagnosis)] = {
            "node": f"{m}_supply",
            "mean_load": np.nan,
            "p95_load": np.nan,
            "max_queue": np.nan,
            "spill_probability": np.nan,
            "storage_ratio": np.nan,
            "resource_type": "supply",
            "resource": m,
            "causal_elasticity": elasticity_m,
            "surface_score": surface_m,
            "role": "root_bottleneck"
            if elasticity_m >= 0.15
            else "secondary_bottleneck",
        }
    return diagnosis.sort_values(
        ["causal_elasticity", "surface_score"], ascending=False
    )


def plot_city(city, df, sensitivity_df, diagnosis):
    """输出队列传播图与因果瓶颈排序图。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for m in MODES:
        axes[0].plot(df["time"], df[f"p_{m}"], label=CN_MODE[m])
    axes[0].set_ylabel("旅客排队人数")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["time"], df["exit_occupancy"], label="离场道路车辆数", lw=2)
    axes[1].plot(
        df["time"],
        df[[f"external_{m}" for m in MODES]].sum(axis=1),
        label="入口外部车辆队列",
    )
    axes[1].set(xlabel="时间（分钟）", ylabel="车辆数")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{CN_CITY[city]}：排队演化与拥堵传播")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_queues.png", dpi=180)
    plt.close(fig)

    top = sensitivity_df.head(10).sort_values("bottleneck_elasticity")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = np.where(top["bottleneck_elasticity"] >= 0, "#2878B5", "#C82423")
    ax.barh(top["resource"].map(cn_resource), top["bottleneck_elasticity"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="因果瓶颈弹性",
        title=f"{CN_CITY[city]}：因果瓶颈弹性排序",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_bottleneck_ranking.png", dpi=180)
    plt.close(fig)

    colors = {
        "root_bottleneck": "#C82423",
        "secondary_bottleneck": "#F39B7F",
        "surface_congestion": "#2878B5",
        "non_bottleneck": "#9E9E9E",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    for role, group in diagnosis.groupby("role"):
        ax.scatter(
            group["surface_score"],
            group["causal_elasticity"],
            s=70,
            color=colors[role],
            label=CN_ROLE[role],
        )
        for _, row in group.iterrows():
            dx, dy = LABEL_OFFSET.get(row["node"], (5, 8))
            ax.annotate(
                cn_resource(row["node"]),
                (row["surface_score"], row["causal_elasticity"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                fontsize=8,
            )
    ax.axvline(0.45, color="grey", ls="--", lw=0.8)
    ax.axhline(0.15, color="grey", ls="--", lw=0.8)
    ax.set(
        xlabel="表面拥堵指数",
        ylabel="因果瓶颈弹性",
        title=f"{CN_CITY[city]}：表面拥堵与因果瓶颈联合诊断",
    )
    ax.margins(x=0.05, y=0.15)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_joint_diagnosis.png", dpi=180)
    plt.close(fig)


def plot_stability(city, stability_summary):
    """绘制多随机种子下的平均瓶颈弹性与标准差。"""
    top = stability_summary.head(8).sort_values("mean_elasticity")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        top["resource"].map(cn_resource),
        top["mean_elasticity"],
        xerr=top["std_elasticity"],
        color="#2878B5",
        alpha=0.9,
        capsize=3,
    )
    ax.set(
        xlabel="平均因果瓶颈弹性（误差线为标准差）",
        title=f"{CN_CITY[city]}：30次随机重复实验瓶颈稳定性",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{city}_stability.png", dpi=180)
    plt.close(fig)


def write_report(summaries, rankings, diagnoses, stabilities):
    lines = [
        "# 问题1计算结果摘要",
        "",
        "瓶颈弹性定义为某项资源增加10%时，系统广义延误成本下降率除以10%。",
        "弹性越大，说明该资源越接近拥堵根因。",
        "",
    ]
    for summary in summaries:
        city = summary["city"]
        top = rankings[city].head(5)
        stable_top = stabilities[city].iloc[0]
        observed = (
            diagnoses[city]
            .query("resource_type == 'physical'")
            .sort_values("surface_score", ascending=False)
            .head(3)
        )
        lines.extend(
            [
                f"## {CN_CITY[city]}",
                "",
                f"- 旅客服务率：{summary['service_rate']:.2%}",
                f"- 平均旅客等待：{summary['avg_passenger_wait_min']:.2f} 分钟",
                f"- 平均车辆等待：{summary['avg_vehicle_wait_min']:.2f} 分钟",
                f"- 最大旅客队列：{summary['max_passenger_queue']:.0f} 人",
                f"- 最大车辆队列：{summary['max_vehicle_queue']:.0f} 辆",
                "- 表面拥堵最明显节点："
                + "、".join(observed["node"].map(cn_resource).tolist()),
                f"- 30次重复实验中排名第一概率最高的资源："
                f"{cn_resource(stable_top['resource'])}"
                f"（{stable_top['top1_probability']:.0%}）",
                "",
                "| 排名 | 资源 | 类型 | 容量增加10%后的成本下降 | 瓶颈弹性 |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(
                f"| {rank} | {cn_resource(row['resource'])} | {CN_TYPE[row['type']]} | "
                f"{row['cost_reduction_pct']:.2f}% | {row['bottleneck_elasticity']:.3f} |"
            )
        lines.append("")
    (OUTPUT / "problem1_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validate_configs()
    export_calibration_parameters()
    summaries, rankings, diagnoses, stabilities = [], {}, {}, {}

    for city in CITIES:
        demand, supply = generate_inputs(city)
        df, base_cost = simulate(city, demand, supply)
        nodes = node_diagnostics(city, df)
        ranking = sensitivity(city, demand, supply, base_cost)
        diagnosis = joint_diagnosis(nodes, ranking)
        summary = system_summary(city, demand, supply, df, base_cost)

        demand.to_csv(OUTPUT / f"{city}_passenger_demand.csv", index=False)
        supply.to_csv(OUTPUT / f"{city}_vehicle_supply.csv", index=False)
        df.to_csv(OUTPUT / f"{city}_simulation.csv", index=False)
        nodes.to_csv(OUTPUT / f"{city}_node_diagnostics.csv", index=False)
        ranking.to_csv(OUTPUT / f"{city}_sensitivity.csv", index=False)
        diagnosis.to_csv(OUTPUT / f"{city}_joint_diagnosis.csv", index=False)
        plot_city(city, df, ranking, diagnosis)
        stability_raw, stability_summary = stability_analysis(city)
        stability_raw.to_csv(OUTPUT / f"{city}_stability_raw.csv", index=False)
        stability_summary.to_csv(OUTPUT / f"{city}_stability_summary.csv", index=False)
        plot_stability(city, stability_summary)

        summaries.append(summary)
        rankings[city] = ranking
        diagnoses[city] = diagnosis
        stabilities[city] = stability_summary

    pd.DataFrame(summaries).to_csv(OUTPUT / "system_summary.csv", index=False)
    write_report(summaries, rankings, diagnoses, stabilities)
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    for city, ranking in rankings.items():
        print(f"\n{city} top bottlenecks:")
        print(ranking.head(5).round(3).to_string(index=False))
    print(f"\nResults written to: {OUTPUT}")


if __name__ == "__main__":
    main()
