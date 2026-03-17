## 一、 技术路线概述

系统分为 **异常检测（Detection）** 与 **工况分类（Classification）** 两个核心阶段：
*   **第一阶段（异常检测）：** 通过多维度时序分析模型，实时监控压力与流量的波动。该阶段采用 **TimesNet** 与 **DADA** 双模型冗余配置，确保告警的灵敏度与真实性。
*   **第二阶段（工况分类）：** 在检测到波动区间后，自动提取波形特征（如峰值、频率、耦合度等），通过模式识别技术判定具体的业务操作类型。

---

## 二、 项目实施进展

### 2.1 异常检测模块

异常检测是系统的基础，目前已完成核心算法的训练与全线多场景回测。

#### 2.1.1 核心算法 A：TimesNet
**技术说明：** TimesNet 是一种先进的时序表征模型，它通过将一维时间序列转化为二维周期图像，利用计算机视觉技术提取复杂工况下的波动特征。

**全覆盖识别验证：**
为证明算法的普适性，我们针对全线关键站点及主要操作类型进行了全量识别验证。

**1. 站点覆盖展示：**
*   **全线相关操作 (1001)：**
    > ![1001_full_diagnosis](./visualization/timesnet_multi_dim_v2/1001_full_diagnosis.png)
*   **鄂托克站 (1002)：**
    > ![1002_full_diagnosis](./visualization/timesnet_multi_dim_v2/1002_full_diagnosis.png)
*   **乌审旗站 (1006)：**
    > ![1006_full_diagnosis](./visualization/timesnet_multi_dim_v2/1006_full_diagnosis.png)
*   **油房庄站 (1027)：**
    > ![1027_full_diagnosis](./visualization/timesnet_multi_dim_v2/1027_full_diagnosis.png)
*   **土默特站 (1042)：**
    > ![1042_full_diagnosis](./visualization/timesnet_multi_dim_v2/1042_full_diagnosis.png)
*   **达拉特站 (1038)：**
    > ![1038_full_diagnosis](./visualization/timesnet_multi_dim_v2/1038_full_diagnosis.png)

**2. 操作类型覆盖展示：**
*   **增量操作 (1010)：** 按计划提量识别。
    > ![1010_full_diagnosis](./visualization/timesnet_multi_dim_v2/1010_full_diagnosis.png)
*   **降量操作 (1007)：** 按计划减量识别。
    > ![1007_full_diagnosis](./visualization/timesnet_multi_dim_v2/1007_full_diagnosis.png)
*   **甩泵异常 (1028)：** 突发泵组故障捕捉。
    > ![1028_full_diagnosis](./visualization/timesnet_multi_dim_v2/1028_full_diagnosis.png)
*   **切泵操作 (1004)：** 正常运行切换。
    > ![1004_full_diagnosis](./visualization/timesnet_multi_dim_v2/1004_full_diagnosis.png)
*   **启停泵 (1005)：** 泵组试运识别。
    > ![1005_full_diagnosis](./visualization/timesnet_multi_dim_v2/1005_full_diagnosis.png)
*   **紧急启停输 (1012)：** 突发停电、水击等重大工况监控。
    > ![1012_full_diagnosis](./visualization/timesnet_multi_dim_v2/1012_full_diagnosis.png)
*   **计划启停输 (1018)：** 计划内作业停输识别。
    > ![1018_full_diagnosis](./visualization/timesnet_multi_dim_v2/1018_full_diagnosis.png)
*   **下载燃料油 (1023)：** 支线作业波动捕捉。
    > ![1023_full_diagnosis](./visualization/timesnet_multi_dim_v2/1023_full_diagnosis.png)

#### 2.1.2 冗余算法 B：DADA
**技术说明：** DADA (Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders) 是一种通用的时间序列异常检测器。它通过**自适应瓶颈（Adaptive Bottlenecks）**和**双对抗解码器（Dual Adversarial Decoders）**架构，实现了零样本（Zero-shot）的异常捕获能力。
*   **作用：** 作为 TimesNet 的冗余备份，DADA 能够独立对流量、压力信号进行敏感度极高的波动捕捉，通过两个模型在异常区间上的“共识”来判定最终告警。

**全覆盖识别验证：**
针对全线关键工况，DADA 展现了与 TimesNet 高度互补的识别能力。

**1. 站点覆盖展示：**
*   **全线相关操作 (1013)：**
    > ![diag_1013](./visualization/splits/diag_1013.png)
*   **鄂托克站 (1015)：**
    > ![diag_1015](./visualization/splits/diag_1015.png)
*   **乌审旗站 (1044)：**
    > ![diag_1044](./visualization/splits/diag_1044.png)
*   **油房庄站 (1067)：**
    > ![diag_1067](./visualization/splits/diag_1067.png)
*   **土默特站 (1049)：**
    > ![diag_1049](./visualization/splits/diag_1049.png)
*   **达拉特站 (1048)：**
    > ![diag_1048](./visualization/splits/diag_1048.png)

**2. 操作类型覆盖展示：**
*   **增量操作 (1021)：**
    > ![diag_1021](./visualization/splits/diag_1021.png)
*   **降量操作 (1022)：**
    > ![diag_1022](./visualization/splits/diag_1022.png)
*   **甩泵异常 (1035)：**
    > ![diag_1035](./visualization/splits/diag_1035.png)
*   **切泵操作 (1036)：**
    > ![diag_1036](./visualization/splits/diag_1036.png)
*   **启停泵 (1061)：**
    > ![diag_1061](./visualization/splits/diag_1061.png)
*   **紧急启停输 (1016)：**
    > ![diag_1016](./visualization/splits/diag_1016.png)
*   **计划启停输 (1019)：**
    > ![diag_1019](./visualization/splits/diag_1019.png)
*   **下载燃料油 (1034)：**
    > ![diag_1034](./visualization/splits/diag_1034.png)

---

## 二、 项目实施进展

### 2.2 工况分类模块

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
