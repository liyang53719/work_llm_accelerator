---
description: "Use when working on Catapult HLS header resolution, local AC/FP compatibility headers, CRD-182 limits.h failures, or CIN-84 top-interface modeling issues in work_llm_accelerator."
applyTo: "hls/**"
---

# Catapult Header Resolution Rules

- `work/` 只允许阅读参考，不能作为真实依赖、输出目录或包含路径来源。
- 本地 AC/FP 兼容头统一放在 `hls/include/`；新增或修改这类头时，不要再散落回 `hls/` 根目录。
- `ac_channel.h` 不属于仓库本地兼容头范畴；必须来自 Catapult 自带的 `Mgc_home/shared/include/ac_channel.h`。
- 任何“在仓库里新建/复制/覆盖 `ac_channel.h`”的方案都视为过时弃用方案，不再考虑，也不要继续传播到新脚本或新文档里。
- Catapult/EDG 遇到 `limits.h`、`climits`、`cmath`、`iostream`、`ac_int.h`、`ac_std_float.h`、`ccs_dw_fp_lib.h` 相关问题时，先确认实际命中的是仓库内本地兼容头，而不是 vendor 或系统头。
- 优先使用显式仓库内包含路径和显式 `.h` shim；不要依赖 `<ac_int.h>` 这类角括号优先级，也不要依赖无扩展名 shim 名字。
- `limits.h` shim 必须保持自包含实现，不要再尝试 `include_next <limits.h>` 这类递归方案。
- `cmath` shim 需要先处理宏污染，再补齐 `std::signbit`、`std::isfinite`、`std::isnormal`、`std::isinf`、`std::isnan`、`std::ceil`、`std::floor` 以及 `std::__builtin_*` 兼容名。
- 如果 `go compile` 出现 `Unable to reduce array size ... (CIN-84)`，优先检查 top wrapper 是否仍在使用无界指针端口；已知稳定做法是改成带明确上界的数组接口。
- 处理完头文件解析问题后，如果失败点前移到 `FPOPS.ccs_fp_*` 或 `incomplete component selection`，说明问题已经进入 operator library/component binding 阶段，不要再回头改系统头链。