## **LongPO：针对长prompt的调优方法**

长prompt调优的难点是：**如何在优化时尽量保证prompt的一致性**。

如果直接将长prompt作为输入，去跑APE方法（核心是rephrase），有以下难点：

1. 将几百字长的prompt直接进行rephrase，对LLM挑战大；
2. 面临不可控问题 -- LLM可能改动的地方太多，导致优化有收敛问题。

因此，这篇文章引入了一个sentence挑选机制：**在每轮优化时，选择一个sentence进行调优**。

其单次迭代的流程如下图所示：

![img](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde554a794fe61e5940614e9e6e7ba7e0e21fbfd941f70666c0ad8a4471a2204992cec177c308ebd5304a43a8ad3bb666bb7ebcc284fa244fe7810037ac2b26631a37c755d2dac63418560f732d969ce23b44fb4c8ed7016461c?tmpCode=f4d6cb47-0c2c-46fa-9244-8855b21b503f)

整个流程的核心点如下：

- 优化单位为sentence；
- 采用`Lin-UCB`算法来选择待优化sentence
  - 这是个`contextual bandit problem`，其中[arms](https://zhida.zhihu.com/search?content_id=237757006&content_type=Article&match_order=1&q=arms&zhida_source=entity)对应m个sentences、每个arm有对应的feature vector、拉下arm带来的reward对应调优后的task metric提升
  - 每个arm的feature vector使用其sentence embedding，原文使用T5-encoder进行表征
- 针对选出的sentence进行prompt optimization
  - **optimizer LLM**：核心方法是 rephrase，直接通过[prompting](https://zhida.zhihu.com/search?content_id=237757006&content_type=Article&match_order=1&q=prompting&zhida_source=entity) LLM实现；同时引入历史正向迭代数据辅助迭代，使用`few-shot In Context Learning`的思路，检索出Top 4语义相似的数据，作为examples。完整prompt如下所示。

![img](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde554a794fe61e5940614e9e6e7ba7e0e21fbfd941f70666c0ad8a4471a2204992cec177c308ebd53048b6ddffd04d3d7c3ec83d573bf13c40fdcc2790f40ed5a7cf4c04dae87bca5bda05f06d98884c5a84fb4c8ed7016461c?tmpCode=f4d6cb47-0c2c-46fa-9244-8855b21b503f)

- **searchprocedure**：由于LLM本身对prompt的细微改动比较敏感，为使整个优化过程更加stable，本法采取`greedy beam search`策略，既**始终保持效果最好的Top K prompts**。
- 重复上述两个步骤 - 选sentence、对sentence进行迭代，直到达到既定的停止条件（如达到iteration轮数）即停止，完成迭代过程。
