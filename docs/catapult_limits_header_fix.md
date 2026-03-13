# Catapult `limits.h` / 标准头解析问题排查记录

## 结论

这次 `CRD-182` 的根因不是 `limits.h` 本身，而是 Catapult/EDG 在 HLS 分析时没有稳定命中仓库内的本地兼容头，导致 `ac_int.h` / `ac_std_float.h` / `ccs_dw_fp_lib.h` 的一部分包含链回退到了 vendor 头和系统标准头，最后落进 `/usr/include/limits.h`，触发：

```text
Error: /usr/include/limits.h(124): could not open source file "limits.h" (no directories in search list) (CRD-182)
```

真正有效的修法不是继续和 glibc 的 `include_next <limits.h>` 纠缠，而是把 Catapult 入口用到的头文件链完整本地化，并且避免使用让 EDG 不稳定的“无扩展名标准头名”解析。

## 现象

最开始的报错出现在 `go analyze`：

```text
Error: /usr/include/limits.h(124): could not open source file "limits.h" (no directories in search list) (CRD-182)
Error: Compilation aborted (CIN-5)
Error: go analyze: Failed analyze
```

这个错误会误导人以为只要补一个 `limits.h` shim 就够了，但实际不是。

## 根因

这次问题里有 4 个关键事实。

1. 只补 `catapult_shims/limits.h` 不够。
2. Catapult 对 `<ac_int.h>`、`<ac_std_float.h>`、`<ccs_dw_fp_lib.h>` 这类角括号包含并不会稳定优先命中仓库内版本。
3. EDG 对无扩展名的 shim 名字也不稳定，比如 `limits`、`cstring`、`cmath`、`iostream` 这类名字会把问题重新带回系统标准库。
4. 真正把问题钉死的是“显式本地路径 + 显式 `.h` shim + 本地化 FP 头链”。

我们后来做的最小复现实验表明：

- 只要 `#include <ac_int.h>`，Catapult 就可能重现 `CRD-182`。
- 但改成显式本地路径 `#include "../../hls/ac_int.h"` 后，`limits.h` 错误立即消失，后续只剩正常的本地 shim 语法/模板问题。

这说明问题根本不在 `limits.h` 文本内容，而在“谁被真正包含了”。

## 这次采用的修法

### 1. 本地化 Catapult 入口头链

HLS 入口不再依赖角括号命中顺序，而是直接显式包含仓库内版本：

- `hls/prefill_only/qwen_prefill_attention_kernel.cpp`
  - `#include "../ac_int.h"`
  - `#include "../ac_std_float.h"`
  - `#include "../ccs_dw_fp_lib.h"`

### 2. 本地化 `ccs_dw_fp_lib.h`

仓库内新增了本地 `hls/ccs_dw_fp_lib.h`，用来覆盖 vendor 版本，并把顶部依赖改成显式本地头：

- `#include "ac_std_float.h"`
- `#include "catapult_shims/iostream.h"`

这样可以保证 FP 库不会再偷偷回到 vendor 的 `<ac_std_float.h>` 或系统的 `<iostream>`。

### 3. 统一改成显式 `.h` shim

这次确认 EDG 对无扩展名 shim 名字不可靠，所以本地 shim 全部补成了显式 `.h` 版本，并让真实包含链只走这些文件：

- `hls/catapult_shims/limits.h`
- `hls/catapult_shims/climits.h`
- `hls/catapult_shims/cmath.h`
- `hls/catapult_shims/cstring.h`
- `hls/catapult_shims/cstdint.h`
- `hls/catapult_shims/cstddef.h`
- `hls/catapult_shims/ostream.h`
- `hls/catapult_shims/iostream.h`
- `hls/catapult_shims/string.h`

不要再依赖下面这种路径：

```cpp
#include <cstring>
#include <cmath>
#include <ac_int.h>
```

在 Catapult 上，这种写法很容易重新掉回 vendor / system 头。

### 4. `limits.h` 不再做系统递归

本地 `hls/catapult_shims/limits.h` 现在是自包含实现，直接提供：

- 整数极值宏
- `std::numeric_limits<整型>`
- `std::numeric_limits<float>`
- `std::numeric_limits<double>`
- `std::numeric_limits<long double>`

核心原则是：

- 不依赖 `include_next <limits.h>`
- 不依赖 GCC `include-fixed/limits.h` 链式回退
- 不再通过中间 `limits` shim 间接跳转

### 5. `cmath` shim 的要求

`ac_std_float.h` 不只是需要 `std::ceil/std::floor`，它还依赖 `std::signbit`、`std::isfinite`、`std::isnormal`、`std::isinf`、`std::isnan` 等名字。

另外，`math.h` 往往会把 `fpclassify/signbit/isnan/isfinite/...` 定义成宏；如果不先 `#undef`，后面的 shim 函数声明会被宏展开直接破坏语法。

所以本地 `hls/catapult_shims/cmath.h` 必须同时做到：

1. 先 `#undef` 这些宏。
2. 提供 `std::fpclassify/signbit/isnan/isfinite/isnormal/isinf/ceil/floor`。
3. 提供 `std::__builtin_signbit/__builtin_isfinite/__builtin_isnormal/__builtin_isinf_sign/__builtin_isnan` 这类 EDG 在 `ac_std_float.h` 里会直接调用的名字。

## TCL 侧的经验

这次最终能越过 `limits.h`，关键不在 TCL 里堆更多系统目录，而在于“源码实际包含的头必须已经完全本地化”。

保留的做法：

- 继续给 Catapult 脚本传 `-D__EDG__`
- `SearchPath` 里仍保留本地 `catapult_shims`、本地 `hls`、vendor `shared/include`、系统 C++ include 目录

不要把希望寄托在下面这些办法上：

- 只补一个 `limits.h` shim
- 只调 `SearchPath` 顺序
- 只加 `-include limits.h`
- 只定义 `_GCC_LIMITS_H_` / `_LIBC_LIMITS_H_`

这些都不能从根上解决“实际命中了错误头文件”的问题。

## 这次验证结果

修复后，`hls/prefill_only/run_catapult_prefill_attention.tcl` 已经可以越过原始的 `go analyze` 阶段，不再出现：

```text
/usr/include/limits.h(124) ... (CRD-182)
```

后续继续排查时，又把 `go compile` 阶段的接口数组建模问题也推进掉了。`CIN-84` 的根因不是算法访问本身，而是 Catapult top wrapper 入口把外部存储声明成了“无界指针”，工具默认按 `1024 words` 建模，导致：

```text
Error: ... Unable to reduce array size for variable 'input_sequence.d', currently 1024 words (CIN-84)
Error: ... go compile: Failed compile
```

这一步的有效修法是：

- 把 Catapult top wrapper 的外部 memory 参数从无界指针改成带明确上界的数组端口。
- 让 `input_sequence`、`q_packed_weights`、`k_cache`、`output_sequence` 这类接口直接暴露最大容量，而不是依赖工具猜测 pointer depth。

修完之后，`go compile` 已经能够完成，日志里不再有 `CIN-84`。

当前新的阻塞点已经前移到 `architect` / component binding，报错表现为：

```text
Warning: Couldn't find library component for operator 'FPOPS.ccs_fp_*' - no available component (SIF-4)
Error: incomplete component selection
```

这说明：

- `limits.h` / 标准头解析问题已经解决。
- 顶层接口的 `CIN-84` 建模问题也已经解决。
- 现在剩下的是下一阶段的 FP operator 库绑定问题，不是系统头问题。

## 后续再遇到同类问题时的操作顺序

1. 不要先改系统 include 链，也不要先改 `/usr/include/limits.h` 的思路。
2. 先做最小复现，确认是 `ac_int`、`ac_std_float` 还是 `ccs_dw_fp_lib` 哪一层触发。
3. 对 Catapult 入口文件使用显式本地相对路径包含，不要依赖 `<ac_int.h>` 这类角括号命中顺序。
4. 把本地 shim 改成显式 `.h` 文件，不要再依赖无扩展名标准头风格名字。
5. `limits.h` 只做自包含定义，不做 `include_next` 递归。
6. `cmath` shim 必须先 `#undef` 宏，再补足 `std::` 名字和 `std::__builtin_*` 名字。
7. 如果 `go compile` 卡在 `Unable to reduce array size ... (CIN-84)`，优先检查 top wrapper 是否还在用无界指针端口。
8. 等 `go compile` 过了以后，再处理后续 assembly / architect 阶段的 operator library 绑定问题。

## 这次直接相关的文件

- `hls/prefill_only/qwen_prefill_attention_kernel.cpp`
- `hls/ccs_dw_fp_lib.h`
- `hls/ac_int.h`
- `hls/ac_float.h`
- `hls/ac_std_float.h`
- `hls/common/llm_accel_types.h`
- `hls/common/llm_layer_dispatch.h`
- `hls/common/llm_memory_layout.h`
- `hls/catapult_shims/limits.h`
- `hls/catapult_shims/climits.h`
- `hls/catapult_shims/cmath.h`
- `hls/catapult_shims/cstring.h`
- `hls/catapult_shims/cstdint.h`
- `hls/catapult_shims/cstddef.h`
- `hls/catapult_shims/ostream.h`
- `hls/catapult_shims/iostream.h`
- `hls/catapult_shims/string.h`
- `hls/prefill_only/run_catapult_prefill_attention.tcl`
- `hls/prefill_only/run_catapult_prefill_mlp.tcl`