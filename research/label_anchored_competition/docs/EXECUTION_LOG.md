# 执行日志

所有时间使用北京时间。每次运行记录命令、源码 commit、协议和输入哈希、设备、日志、退出码与输出清单；不能将自检记作研究结果。

## 2026-10-20：方向切换与执行前记录

已完整读取用户指定的“论文拒稿修改建议”最新两轮讨论，采用标签锚定的竞争类别验证，按冻结诊断、训练收益、泛化成本顺序执行。旧 RCV 九组结果保留，不作为新机制结果。

GitHub main 的基线 commit 为 `181f1472109f28856c199b08c4b3b12a0e3bcb5d`，与此前冻结诊断所用版本一致。新分支为 `research/label-anchored-competition`，计划与协议先提交同步，之后执行代码和数据检查。

已核验 GitHub SSH 身份可用。服务器旧入口的代理域名当前 DNS 解析失败；已向用户请求可用入口，未启动新教师或端到端训练。新实例的实时 GPU 数量、VOC 训练图像是否齐全、可用预训练权重及环境均待连通后审计。

本地保留有原 VOC 1/8 划分、增强标注压缩包和预训练权重，可先进行数据隔离与标注支持量审计。标签预算是 1,323 张，不与 DINOv2 分支的 183-label 设置混用。

## 2026-10-20：执行 P0 数据审计和本地自检

执行前计划已在 commit `aa2d3eff` 推送至研究分支。随后按协议处理真实的原 VOC 1/8 列表与增强标注压缩包，原始输入哈希均匹配。新分区为 D_sup=1,191、D_rel=132、D_U=9,259、validation=1,449，六项两两图像交集均为零。D_sup 的 21 类均有支持，最少为 class10 的 23 张图；D_rel 最少为 class3 的 2 张图，后续每类正确性分析须如实报告支持不足。未据这些支持量重新挑划分。

合成数值／隔离检查通过 16/16、零跳过，覆盖极高 MC、独立 NumPy 参考、残余类别置换与均匀期望、同一像素重组 batch、特征对齐、缺类／零范数失败、detach 权重和全有效像素损失分母。环境为本地 CPU、PyTorch 2.5.1；不能把它记作服务器 CUDA 自检、真实训练 smoke 或新机制的实证收益。

复现数据审计：`python -m research.label_anchored_competition.prepare_data --split-root /path/to/splits/pascal --mask-archive /path/to/SegmentationClass.zip --output /new/output`。自检：`python -m research.label_anchored_competition.run_selfcheck --output /new/selfcheck.json`。输出目录和收据禁止覆盖旧结果。具体输入／源码／输出哈希见 records。

## 2026-10-20：服务器恢复连通

按用户确认重试后，原入口恢复可用。当前设备为 4 张空闲 RTX 4090，基础 PyTorch 2.2.2/CUDA 12.1；与前一阶段的 H100 环境不同，后续记录当前硬件，不声称物理设备相同。旧持久结果完整，原临时环境为空。

为新研究建立独立持久目录，使用此前 wheel 重建 Lightning 2.0.9、scikit-learn 1.5.2 等依赖。现存恢复包只有 1,449 张 VOC validation 图像，故从官方 Oxford 归档补齐训练图像，预先固定归档字节数 1,999,639,040、MD5 `6cd6e144f989b92b3379bac3b3de84fd`；下载后才解压，并用原增强标注覆盖训练掩码、从官方归档恢复 validation 掩码。下载、环境与数据准备均独立记录。

正式教师入口已准备，保留原 SSL 损失和增强，限制主模型输入为 D_sup/D_U。新入口把原手动 LR scheduler 注册到 Lightning，避免恢复时丢状态；按固定 global step 保存，验证集填充行置为 ignore 避免 DDP 重复计数。服务器 smoke 通过前不标记正式教师 running。
