# 气液管道运行工况识别与异常诊断项目阶段性汇报

**汇报日期：** 2026年1月18日
**项目目标：** 构建具备工业级可靠性的异常监测与诊断系统，实现全线工况的**自动识别**、**冗余检测**与**精准分类**。

---

## 一、 总体技术架构

本项目采用“二级分层、双模检测”的工业级架构，旨在建立从底层异常捕捉到高层业务决策的完整链路。

### 1. 技术路线概述
系统分为**异常检测（Detection）**与**工况分类（Classification）**两个核心阶段：
*   **第一阶段（异常检测）：** 通过多维度时序分析模型，实时监控压力与流量的波动。该阶段采用 **TimesNet** 与 **DADA** 双模型冗余配置，确保告警的灵敏度与真实性。
*   **第二阶段（工况分类）：** 在检测到波动区间后，自动提取波形特征（如峰值、频率、耦合度等），通过模式识别技术判定具体的业务操作类型。

### 2. 核心设计理念
*   **单维度精细建模：** 针对关键站点（如乌审旗、鄂托克）的独立物理指标进行建模，避免全局数据中的“异常信号稀释”。
*   **冗余共识机制：** 只有当冗余检测模型达成一致时，系统才输出确定性结论，最大限度压低虚警率。

---

## 二、 项目实施进展

### 2.1 异常检测模块

异常检测是系统的基础，目前已完成核心算法的训练与全线多场景回测。

#### 2.1.1 核心算法 A：TimesNet (已完成)
**技术说明：** TimesNet 是一种先进的时序表征模型，它通过将一维时间序列转化为二维周期图像，利用计算机视觉技术提取复杂工况下的波动特征。

**全覆盖识别验证：**
为证明算法的普适性，我们针对全线关键站点及主要操作类型进行了全量识别验证。

**1. 站点覆盖展示：**
*   **全线相关操作 (1001)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1001_full_diagnosis.png]
*   **鄂托克站 (1002)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1002_full_diagnosis.png]
*   **乌审旗站 (1005)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1005_full_diagnosis.png]
*   **油房庄站 (1027)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1027_full_diagnosis.png]
*   **土默特站 (1036)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1036_full_diagnosis.png]
*   **达拉特站 (1048)：**
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1048_full_diagnosis.png]

**2. 操作类型覆盖展示：**
*   **增量操作 (1010)：** 按计划提量识别。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1010_full_diagnosis.png]
*   **降量操作 (1007)：** 按计划减量识别。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1007_full_diagnosis.png]
*   **甩泵异常 (1028)：** 突发泵组故障捕捉。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1028_full_diagnosis.png]
*   **切泵操作 (1004)：** 正常运行切换。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1004_full_diagnosis.png]
*   **启停泵 (1005)：** 泵组试运识别。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1005_full_diagnosis.png]
*   **紧急启停输 (1012)：** 突发停电、水击等重大工况监控。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1012_full_diagnosis.png]
*   **计划启停输 (1018)：** 计划内作业停输识别。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1018_full_diagnosis.png]
*   **下载燃料油 (1023)：** 支线作业波动捕捉。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/operating-condition-time-series/TSLib-Custom/vis_results/multi_dim_v2/1023_full_diagnosis.png]

#### 2.1.2 冗余算法 B：DADA (已完成)
**技术说明：** DADA (Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders) 是一种通用的时间序列异常检测器。它通过**自适应瓶颈（Adaptive Bottlenecks）**和**双对抗解码器（Dual Adversarial Decoders）**架构，实现了零样本（Zero-shot）的异常捕获能力。
*   **作用：** 作为 TimesNet 的冗余备份，DADA 能够独立对流量、压力信号进行敏感度极高的波动捕捉，通过两个模型在异常区间上的“共识”来判定最终告警。

**DADA 识别结果展示：**
我们选取了部分典型工况，展示 DADA 在不同文件及传感器（CHX00F003FT0101 & CHX00F002FT0101）下的检测表现：

*   **突发波动识别 (1013)：** 精准捕捉流量的阶跃式变化。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1013.png]
*   **复杂震荡监控 (1017)：** 在压力与流量同步震荡时，准确锁定高分区间。
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1017.png]
*   **甩泵/计划停输识别 (1018/1028)：** 
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1018.png]
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1028.png]
*   **下载作业波动 (1023)：** 
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1023.png]
*   **长期运行稳定性 (1039)：** 
    > ![/Users/liuqiyuan/Documents/项目/operating-condition-time-series/DADA/visualization/splits/diag_1039.png]

---

### 2.2 冗余检测共识机制 (核心优势)
系统通过对比 **TimesNet** 的能量重构分数与 **DADA** 的自适应瓶颈分数，寻找交集区间（Joint Anomaly）。
*   **双重验证：** 如图中红色阴影区域所示，当两条独立的技术路线同时触发预警时，系统可靠性提升至工业级标准。
*   **独立度量：** 两个模型采用不同的归一化与特征提取逻辑，有效规避了单模型的系统性偏见。

---

### 2.3 工况分类模块

工况分类旨在实现从“发现异常”到“解释原因”的转化。

#### 2.2.1 异常区间提取逻辑
算法实时监控重构能量指标，当能量超过预设阈值时，自动提取该波动区间的时间切片。
*   **当前成果：** 已实现波形片段的自动化提取。

#### 2.2.2 自动分类识别 (待开发)
**技术路径：** 利用提取的波形特征，构建标准工况指纹库。
*   **分类目标：**
    *   **计划内操作识别：** 增减量、切泵、启停泵、启停输、下载燃料油等。
    *   **异常故障预警：** 故障甩泵、压力波突变、流量漂移等。

> **[此处预留未来分类模型推理逻辑展示图]**

---

**汇报总结：**
项目目前已建立起完整的冗余检测技术框架，**TimesNet** 与 **DADA** 双模型已成功上线并实现对全线关键工况的精准捕捉与互校验证。下一步将重点推进**区间自动分类模型**的开发，最终交付一套工业级的闭环监测决策系统。
