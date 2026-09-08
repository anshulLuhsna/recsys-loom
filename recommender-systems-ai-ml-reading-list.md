# Modern Recommender Systems with AI/ML/LLMs: A Reading List for Myntra Shoes

Verified 26 August 2026. This is a comprehensive but curated path through modern recommender-system design, not a claim to cover the entire literature. Sources are restricted to original papers, first-party engineering blogs, and official technical documentation. Company posts describe what those companies report; they are not independent replications.

## How to use this list

For every reading, capture five things:

```markdown
## Claim

## Mechanism

## Constraint or failure mode

## Decision this changes in Myntra Shoes

## Smallest experiment I can run
```

Do not begin by implementing the most advanced model here. First build a deterministic retrieval and ranking baseline, define hard catalog constraints, and create a fixed set of realistic shoe-discovery queries. The readings should help you justify changes against that baseline.

## Short must-read path

Read these 12 in order. Together they move from the basic funnel to production retrieval, multimodal commerce, sequential personalization, LLM integration, evaluation, and marketplace responsibility.

1. **F1 — Matrix Factorization Techniques for Recommender Systems:** learn what collaborative signals can and cannot represent.
2. **R1 — Deep Neural Networks for YouTube Recommendations:** internalize candidate generation versus ranking.
3. **F8 — DCN V2:** understand feature interactions in a practical web-scale ranker.
4. **R7 — Embedding-based retrieval for Airbnb search:** see how training labels, ANN choice, filters, freshness, and latency interact.
5. **K2 — Deep Learning for Search Ranking at Etsy:** study the migration from trees to a neural commerce ranker.
6. **P1 — Bringing Personalized Search to Etsy:** connect clicks, favorites, carts, purchases, user vectors, privacy, and serving.
7. **S5 — PinnerFormer:** compare long-term sequence representations with real-time state.
8. **M1 — Pinterest Hybrid Search:** design text-plus-image shoe discovery without confusing retrieval and final ranking.
9. **R9 — LinkedIn Hiring Assistant semantic search:** study LLM labels, dual-tower distillation, ANN, hard filters, ranking, freshness, and evaluation as one system.
10. **G1 — Netflix GenRec:** see what an LLM-native ranker changes—and what catalog constraints and serving costs still require.
11. **E1 — Airbnb interleaving:** learn why offline ranking gains are not enough.
12. **E5 — Joint Multisided Exposure Fairness:** treat buyers, products, brands, and sellers as different stakeholders in a marketplace.

## Extended catalog

### A. Foundations: objectives, implicit feedback, and feature interaction

- **F1. [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263)** — *Paper*; Yehuda Koren, Robert Bell, and Chris Volinsky; 2009. **System layer:** collaborative-filtering foundation. **Why read:** It gives you the latent-factor baseline needed to explain what shoe co-view and co-purchase behavior captures that product text and images do not.

- **F2. [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22)** — *Paper*; Yifan Hu, Yehuda Koren, and Chris Volinsky; 2008. **System layer:** data and objective design. **Why read:** Shoe systems mostly observe views, clicks, carts, and purchases rather than ratings, so you need its distinction between preference and confidence before treating every unclicked impression as dislike.

- **F3. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)** — *Paper*; Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme; 2009 conference paper, arXiv version 2012. **System layer:** pairwise ranking objective. **Why read:** It provides a clean first learned-ranking objective for putting a purchased or clicked shoe above sampled alternatives.

- **F4. [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)** — *Paper*; Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua; 2017. **System layer:** interaction modeling. **Why read:** It shows what changes when a neural interaction function replaces a simple user-item dot product, giving you a measured upgrade path from matrix factorization.

- **F5. [Wide & Deep Learning for Recommender Systems](https://research.google/pubs/wide-deep-learning-for-recommender-systems/)** — *Paper*; Heng-Tze Cheng et al., Google; 2016. **System layer:** ranking architecture. **Why read:** Shoe discovery needs both memorization of exact crosses such as brand-plus-category and generalization to sparse combinations, which is the central design tension of this model.

- **F6. [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)** — *Paper*; Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He; 2017. **System layer:** feature interactions and ranking. **Why read:** It offers a practical way to learn low- and high-order crosses among price, brand, size, category, user, query, and context without hand-authoring every interaction.

- **F7. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)** — *Paper*; Maxim Naumov et al., Facebook/Meta; 2019. **System layer:** large-scale ranker architecture. **Why read:** DLRM makes the systems cost of huge sparse embedding tables concrete, helping you separate an educational prototype from architecture that only makes sense at marketplace scale.

- **F8. [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)** — *Paper*; Ruoxi Wang et al., Google; 2020/WWW 2021. **System layer:** ranking and explicit feature crossing. **Why read:** It is a strong bridge from textbook models to a realistic shoe ranker where query-product, user-product, and context-product interactions matter.

### B. Candidate generation, embeddings, two towers, and ANN retrieval

- **R1. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)** — *Paper*; Paul Covington, Jay Adams, and Emre Sargin, Google/YouTube; 2016. **System layer:** candidate generation and ranking. **Why read:** This is the clearest starting point for the two-stage funnel you should reproduce: retrieve broadly from the shoe catalog, then spend more computation ranking a smaller set.

- **R2. [TensorFlow Recommenders: Basic Retrieval](https://www.tensorflow.org/recommenders/examples/basic_retrieval)** — *Official documentation/tutorial*; TensorFlow; continuously maintained, verified 2026. **System layer:** two-tower retrieval implementation. **Why read:** It turns query and candidate towers, in-batch negatives, top-k retrieval, and evaluation into a small implementation you can rebuild by hand before using the library.

- **R3. [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396)** — *Paper*; Ruiqi Guo et al., Google; 2019/ICML 2020. **System layer:** approximate nearest-neighbor serving. **Why read:** It explains why vector-index compression should preserve ranking-relevant inner products rather than merely reconstruct embeddings, a key tradeoff once brute-force shoe retrieval becomes too slow.

- **R4. [Introducing Pixie, an advanced graph-based recommendation system](https://medium.com/pinterest-engineering/introducing-pixie-an-advanced-graph-based-recommendation-system-e7b4229b664b)** — *Engineering article*; Pong Eksombatchai and Mark Ulrich, Pinterest; 31 March 2017. **System layer:** real-time graph candidate generation. **Why read:** Pixie shows a non-neural retrieval path based on weighted random walks, useful for testing whether shoe co-save or co-view graphs beat embeddings for related-item discovery.

- **R5. [PinSage: A new graph convolutional neural network for web-scale recommender systems](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)** — *Engineering article*; Pinterest Engineering; 6 June 2018. **System layer:** graph representation learning and retrieval. **Why read:** It combines visual, text, and graph context to disambiguate visually similar objects—the same problem you face when two shoes look alike but serve different uses.

- **R6. [Establishing a Large Scale Learned Retrieval System at Pinterest](https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5)** — *Engineering article*; Bowen Deng et al., Pinterest; 2025. **System layer:** learned candidate generation and pre-ranking. **Why read:** It shows how a company moves from heuristic graph sources toward engagement-trained embeddings while preserving a multi-stage funnel.

- **R7. [Embedding-based retrieval for Airbnb search](https://medium.com/airbnb-engineering/embedding-based-retrieval-for-airbnb-search-aabebfc85839)** — *Engineering article*; Huiji Gao et al., Airbnb; 19 March 2025. **System layer:** two-tower retrieval and ANN serving. **Why read:** Its decisions about hard negatives, offline item embeddings, IVF versus HNSW, filters, and rapidly changing availability map directly to price, size, and stock in Myntra Shoes.

- **R8. [Innovative Recommendation Applications Using Two Tower Embeddings at Uber](https://www.uber.com/en-BE/blog/innovative-recommendation-applications-using-two-tower-embeddings/)** — *Engineering article*; Uber Engineering; 26 July 2023. **System layer:** reusable two-tower embeddings. **Why read:** It shows why shared user and item representations can power several surfaces and how a global model can replace fragmented local models.

- **R9. [Semantic Search for AI Agents at Scale: Retrieval and Ranking for LinkedIn’s Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant)** — *Engineering article*; Nikita Zhiltsov, Achyuthan Jootoo Ramesh Bapu, Nikhil Thakur, and Dan Liu, LinkedIn; 11 June 2026. **System layer:** LLM-supervised dual-tower retrieval, ranking, freshness, and evaluation. **Why read:** This is an unusually complete account of distilling expensive LLM judgments into embeddings, combining ANN with hard filters, reusing embeddings in ranking, and updating a billion-document index.

- **R10. [From Clicks to Conversions: Architecting Shopping Conversion Candidate Generation at Pinterest](https://medium.com/pinterest-engineering/from-clicks-to-conversions-architecting-shopping-conversion-candidate-generation-at-pinterest-04cae5e1455b)** — *Engineering article*; Richard Huang, Yu Liu, Ziwei Guo, Andy Mao, and Supeng Ge, Pinterest; 27 April 2026. **System layer:** ecommerce candidate generation and multi-task learning. **Why read:** It directly confronts sparse, noisy purchase labels and combines context, sequence, graph, and multimodal product signals across shopping surfaces.

### C. Ranking, pre-ranking, reranking, and multi-objective slates

- **K1. [TensorFlow Recommenders: Basic Ranking](https://www.tensorflow.org/recommenders/examples/basic_ranking)** — *Official documentation/tutorial*; TensorFlow; continuously maintained, verified 2026. **System layer:** pointwise ranking implementation. **Why read:** It is a compact reference for building your first user-item scorer and for seeing exactly how ranking differs from retrieval.

- **K2. [Deep Learning for Search Ranking at Etsy](https://www.etsy.com/codeascraft/deep-learning-for-search-ranking-at-etsy)** — *Engineering article*; Lucia Yu, Congzhe Su, Cung Tran, and Robert Forgione, Etsy; 4 October 2022. **System layer:** second-pass neural ranking. **Why read:** Etsy documents the data, infrastructure, and experimentation work needed to replace a strong tree model, keeping you from mistaking a new architecture for an automatic product improvement.

- **K3. [Improving the Quality of Recommended Pins with Lightweight Ranking](https://medium.com/pinterest-engineering/improving-the-quality-of-recommended-pins-with-lightweight-ranking-8ff5477b20e3)** — *Engineering article*; Poorvi Bhargava, Sen Wang, Andrew Liu, and Duo Zhang, Pinterest; 10 September 2020. **System layer:** pre-ranking. **Why read:** It explains why a cheap personalized scorer belongs between high-recall retrieval and an expensive final ranker, and how much precision you can trade for throughput.

- **K4. [Modernizing Home Feed Pre-Ranking Stage](https://medium.com/pinterest-engineering/modernizing-home-feed-pre-ranking-stage-e636c9cdc36b)** — *Engineering article*; James Li et al., Pinterest; 2025. **System layer:** unified pre-ranking. **Why read:** It shows how source-by-source lightweight rankers become a system-level bottleneck and motivates a single pre-ranking layer across candidate sources.

- **K5. [Learning to rank diversely](https://medium.com/airbnb-engineering/learning-to-rank-diversely-add6b1929621)** — *Engineering article*; Malay Haldar, Liwei He, and Moose Abdool, Airbnb; 30 January 2023. **System layer:** set-level reranking and diversity. **Why read:** A list of ten nearly identical black sneakers is poor discovery even if every item scores well individually; this article shows how to optimize the collection, not only each shoe.

- **K6. [Evolution of Multi-Objective Optimization at Pinterest Home Feed](https://medium.com/pinterest-engineering/evolution-of-multi-objective-optimization-at-pinterest-home-feed-06657e33cd10)** — *Engineering article*; Jiacong He et al., Pinterest; 7 April 2026. **System layer:** final reranking and business objectives. **Why read:** It makes explicit how the last stage balances immediate actions, longer-term outcomes, new use cases, and business rules instead of pretending there is one universal relevance score.

- **K7. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)** — *Engineering article*; Mihajlo Grbovic et al., Airbnb; 5 February 2019. **System layer:** ranking-system evolution. **Why read:** It demonstrates how model and infrastructure complexity should grow with inventory and evidence, which is the right discipline for a from-scratch shoe project.

### D. Personalization, feedback, exploration, and long-term value

- **P1. [Bringing Personalized Search to Etsy](https://www.etsy.com/codeascraft/bringing-personalized-search-to-etsy)** — *Engineering article*; Etsy Search team; 2020. **System layer:** personalized search ranking. **Why read:** It connects multiple feedback strengths, several embedding types, time windows, privacy, cache loss, and cold start in a real product marketplace.

- **P2. [For Your Ears Only: Personalizing Spotify Home with Machine Learning](https://engineering.atspotify.com/2020/1/for-your-ears-only-personalizing-spotify-home-with-machine-learning)** — *Engineering article*; Spotify Engineering; 9 January 2020. **System layer:** exploration and online personalization. **Why read:** It gives you a production example of balancing exploitation with exploration instead of permanently reinforcing the shoes that happened to get early clicks.

- **P3. [Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology](https://research.google/pubs/reinforcement-learning-for-slate-based-recommender-systems-a-tractable-decomposition-and-practical-methodology/)** — *Paper*; Eugene Ie et al., Google/YouTube; 2019. **System layer:** slate recommendation and long-term value. **Why read:** SlateQ shows why optimizing a page of products over time is a different problem from predicting the next click on one item.

- **P4. [Values of Exploration in Recommender Systems](https://research.google/pubs/values-of-exploration-in-recommender-systems/)** — *Paper*; Can Xu et al., Google; RecSys 2021. **System layer:** exploration policy and quality metrics. **Why read:** It gives exploration concrete outcomes—accuracy, diversity, novelty, and serendipity—that you can measure on shoe result lists.

- **P5. [Reward Shaping for User Satisfaction in a REINFORCE Recommender](https://research.google/pubs/reward-shaping-for-user-satisfaction-in-a-reinforce-recommender/)** — *Paper*; Konstantina Christakopoulou et al., Google; 2021. **System layer:** satisfaction-aware learning. **Why read:** It warns that clicks are not satisfaction and shows one way to combine sparse survey-like signals with dense interaction logs.

- **P6. [Surrogate for Long-Term User Experience in Recommender Systems](https://research.google/pubs/surrogate-for-long-term-user-experience-in-recommender-systems/)** — *Paper*; Can Xu et al., Google; KDD 2022. **System layer:** long-term objective design. **Why read:** It teaches you to validate short-term behavioral proxies against a longer-term outcome rather than casually calling engagement “user value.”

### E. Sequence and session recommenders

- **S1. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)** — *Paper*; Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk; 2015/2016. **System layer:** anonymous-session modeling. **Why read:** GRU4Rec is the foundational argument for using the order of a browsing session when persistent user history is absent or weak.

- **S2. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)** — *Paper*; Wang-Cheng Kang and Julian McAuley; 2018. **System layer:** next-item sequence modeling. **Why read:** SASRec gives you a compact Transformer baseline for deciding which prior shoe interactions matter to the next recommendation.

- **S3. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)** — *Paper*; Fei Sun et al., Alibaba; 2019. **System layer:** masked sequential modeling. **Why read:** It offers a bidirectional alternative to next-item prediction and helps you reason about whether order is strict or whether surrounding session context can be used during training.

- **S4. [Towards Neural Mixture Recommender for Long Range Dependent User Sequences](https://research.google/pubs/towards-neural-mixture-recommender-for-long-range-dependent-user-sequences/)** — *Paper*; Jiaxi Tang et al., Google/YouTube; WWW 2019. **System layer:** short- and long-range preference modeling. **Why read:** Shoe intent can change within minutes while style and brand preferences persist for years; M3 treats those temporal ranges as distinct signals.

- **S5. [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/abs/2205.04507)** — *Paper*; Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg, Pinterest; 2022. **System layer:** long-term user representation. **Why read:** It is especially useful for its production choice to approximate real-time sequence quality with daily batch embeddings and a long-horizon objective.

- **S6. [How Pinterest Leverages Realtime User Actions in Recommendation to Boost Homefeed Engagement Volume](https://medium.com/pinterest-engineering/how-pinterest-leverages-realtime-user-actions-in-recommendation-to-boost-homefeed-engagement-volume-165ae2e8cde8)** — *Engineering article*; Xue Xia et al., Pinterest; 4 November 2022. **System layer:** real-time sequence features and GPU ranking. **Why read:** It shows what you gain—and what serving cost you accept—when the latest clicks and hides enter the ranker immediately.

- **S7. [Next-level personalization: How 16K lifelong user actions supercharge Pinterest’s recommendations](https://medium.com/pinterest-engineering/next-level-personalization-how-16k-lifelong-user-actions-supercharge-pinterests-recommendations-bd5989f8f5d3)** — *Engineering article*; Xue Xia et al., Pinterest; 2025. **System layer:** very-long-history sequence modeling. **Why read:** It pushes the context-length question to an industrial extreme and gives you a counterpoint to keeping only a short recent shoe session.

### F. Multimodal and ecommerce product discovery

- **M1. [Hybrid Search: Building a textual and visual discovery experience at Pinterest](https://medium.com/pinterest-engineering/hybrid-search-building-a-textual-and-visual-discovery-experience-at-pinterest-8527ba9728a9)** — *Engineering article*; Matthew Fong, Pinterest; 6 May 2019. **System layer:** text-image retrieval and ranking. **Why read:** It is the closest architectural match to “find shoes like this image, but suitable for this textual intent,” including separate retrieval, lightweight scoring, and relevance scoring.

- **M2. [Introducing Complete the Look: a scene-based complementary recommendation system](https://medium.com/pinterest-engineering/introducing-complete-the-look-a-scene-based-complementary-recommendation-system-eb891c3fe88)** — *Engineering article*; Pinterest Engineering; 2019. **System layer:** visual complementary recommendation. **Why read:** It distinguishes visually similar products from products that complete an outfit, opening a richer direction than another “similar sneakers” demo.

- **M3. [MAPS: Multimodal Attention for Product Similarity](https://www.amazon.science/publications/maps-multimodal-attention-for-product-similarity)** — *Paper*; Amazon researchers; 2021. **System layer:** catalog representation and product similarity. **Why read:** It shows how raw images and catalog attributes can be fused for scalable product representations when either modality alone is incomplete.

- **M4. [Unsupervised Multi-Modal Representation Learning for High Quality Retrieval of Similar Products at E-commerce Scale](https://www.amazon.science/publications/unsupervised-multi-modal-representation-learning-for-high-quality-retrieval-of-similar-products-at-e-commerce-scale)** — *Paper*; Kushal Kumar et al., Amazon; 2023. **System layer:** multimodal product embeddings and nearest-neighbor retrieval. **Why read:** It gives you an ecommerce route to image-plus-text product vectors without requiring manually labeled shoe pairs.

- **M5. [MERLIN: Multimodal & Multilingual Embedding for Recommendations at Large-scale via Item Associations](https://www.amazon.science/publications/merlin-multimodal-multilingual-embedding-for-recommendations-at-large-scale-via-item-associations)** — *Paper*; Sambeet Tiady et al., Amazon; CIKM 2024. **System layer:** graph, multimodal, multilingual product recommendation. **Why read:** MERLIN addresses cold-start products, asymmetric item relationships, multiple languages, co-view graphs, and catalog metadata in one commerce-specific design.

- **M6. [Towards Unified Multi-modal Personalization: Large Vision-Language Models for Generative Recommendation and Beyond](https://www.amazon.science/publications/towards-unified-multi-modal-personalization-large-vision-language-models-for-generative-recommendation-and-beyond)** — *Paper*; Tianxin Wei et al., Amazon and collaborators; ICLR 2024. **System layer:** vision-language personalization. **Why read:** It explores a shared generative framework across recommendation, product search, preference prediction, explanation, and image generation—the ambitious end state for the shoe project.

- **M7. [A Zero Attention Model for Personalized Product Search](https://www.amazon.science/publications/a-zero-attention-model-for-personalized-product-search)** — *Paper*; Qingyao Ai, Daniel N. Hill, S. V. N. Vishwanathan, and W. Bruce Croft; CIKM 2019. **System layer:** query-aware personalization. **Why read:** It tackles when purchase history should influence a product query and when the current query should override history, a crucial guard against irrelevant over-personalization.

- **M8. [Multimodal Learning with Online Text Cleaning for E-commerce Product Search](https://www.amazon.science/publications/multimodal-learning-with-online-text-cleaning-for-e-commerce-product-search)** — *Paper*; Zhizhang Hu, Shasha Li, Ming Du, Arnab Dhua, and Douglas Gray; 2024. **System layer:** noisy catalog text and multimodal search. **Why read:** Marketplace titles and descriptions are messy; this gives you a concrete design for using images while cleaning seller-provided text during training.

### G. Search-recommendation convergence and LLM/generative recommendation

- **G1. [GenRec: Towards LLM-Native Recommendation at Netflix](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3)** — *Engineering article*; Ying Li, Arjun Rao, and Shradha Sehgal, Netflix; 30 July 2026. **System layer:** LLM-backed full-catalog ranking. **Why read:** It is the best current production account of verbalizing histories, using a catalog-aware scoring head, reward-weighted objectives, prefill-only inference, context compaction, and online A/B validation.

- **G2. [Building the agentic future of recruiting: how we engineered LinkedIn’s Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant)** — *Engineering article*; Xiaoyang Gu, Xie Lu, and Daniel Hewlett, LinkedIn; 21 October 2025. **System layer:** agent orchestration, personalization, tools, and human oversight. **Why read:** It shows how an agent can clarify intent, invoke specialized search and evaluation tools, learn from explicit and implicit feedback, and still keep consequential decisions under human control—a useful outer architecture for conversational shoe discovery.

- **G3. [How Etsy Uses LLMs to Improve Search Relevance](https://www.etsy.com/codeascraft/how-etsy-uses-llms-to-improve-search-relevance)** — *Engineering article*; Yuqing Zhang, Congzhe Su, and Susan Liu, Etsy; 16 January 2026. **System layer:** LLM labeling, semantic relevance, and search ranking. **Why read:** Etsy shows a practical use of LLMs as scalable teachers and evaluators, complementing biased engagement labels rather than replacing the serving stack with generation.

- **G4. [Recommendation as Language Processing (P5)](https://arxiv.org/abs/2203.13366)** — *Paper*; Shijie Geng et al.; 2022. **System layer:** unified text-to-text recommendation. **Why read:** P5 is a foundational attempt to express several recommendation tasks through prompts, useful for understanding both the appeal and the awkwardness of turning IDs and interactions into language tokens.

- **G5. [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation](https://arxiv.org/abs/2305.00447)** — *Paper*; Keqin Bao et al.; 2023. **System layer:** parameter-efficient LLM adaptation. **Why read:** It gives you a small-data experiment for testing whether recommendation-specific tuning adds value beyond prompting, without pretending a general LLM already understands your catalog.

- **G6. [LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking](https://arxiv.org/abs/2311.02089)** — *Paper*; Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, and Even Oldridge; 2023. **System layer:** LLM reranking. **Why read:** Its small-retriever-plus-LLM-ranker design is a realistic prototype boundary and avoids expensive free-form generation for every catalog item.

- **G7. [CALRec: Contrastive Alignment of Generative LLMs For Sequential Recommendation](https://research.google/pubs/calrec-contrastive-alignment-of-generative-llms-for-sequential-recommendation/)** — *Paper*; Yaoyiran Li et al., Google; RecSys 2024. **System layer:** generative and contrastive sequential recommendation. **Why read:** It connects two-tower representation learning with generation instead of forcing you to choose one paradigm for user and item understanding.

- **G8. [Factual and Personalized Recommendation Language Modeling with Reinforcement Learning](https://research.google/pubs/factual-and-personalized-recommendation-language-modeling-with-reinforcement-learning/)** — *Paper*; Jihwan Jeong et al., Google; COLM 2024. **System layer:** conversational explanations and factuality. **Why read:** It treats factual consistency, personalization, and persuasiveness as separate rewards, exactly the distinction needed for honest “why this shoe fits” explanations.

- **G9. [LLMRec: Benchmarking Large Language Models on Recommendation Task](https://arxiv.org/abs/2308.12241)** — *Paper*; Junling Liu et al.; 2023. **System layer:** LLM recommendation evaluation. **Why read:** Its results are a useful brake on hype: LLMs were only moderate on accuracy-oriented recommendation tasks while being stronger at explanation-oriented tasks.

### H. Production serving, freshness, feature consistency, and system architecture

- **O1. [Meet Michelangelo: Uber’s Machine Learning Platform](https://www.uber.com/us/en/blog/michelangelo-machine-learning-platform/)** — *Engineering article*; Jeremy Hermann and Mike Del Balso, Uber; 5 September 2017. **System layer:** feature store, training, and online serving. **Why read:** It explains batch versus near-real-time features and training-serving consistency, giving you a concrete architecture for stock, price, popularity, and user-history freshness.

- **O2. [Building a dynamic and responsive Pinterest](https://medium.com/pinterest-engineering/building-a-dynamic-and-responsive-pinterest-7d410e99f0a9)** — *Engineering article*; Bo Liu, Pinterest; 20 September 2018. **System layer:** online candidate generation, feature serving, and ranking. **Why read:** It is a system-design account of replacing stale pregenerated feeds with live graph, candidate, feature, and low-latency ranking services.

- **O3. [Manas Realtime — Enabling changes to be searchable in a blink of an eye](https://medium.com/pinterest-engineering/manas-realtime-enabling-changes-to-be-searchable-in-a-blink-of-an-eye-36acc3506843)** — *Engineering article*; Michael Mi, Pinterest; 13 January 2021. **System layer:** index freshness and recovery. **Why read:** It forces you to design how a price, size, or stock update becomes searchable, how segments compact, and how the system recovers from missed updates.

- **O4. [In-House LLM Serving at Netflix](https://netflixtechblog.com/in-house-llm-serving-at-netflix-a5a8e799ea2c)** — *Engineering article*; Netflix AI Platform Model Runtime and Inference teams; 17 July 2026. **System layer:** GPU inference serving. **Why read:** GenRec only becomes a system when batching, model packaging, rollout, observability, caching, and prefill-only ranking work under load; this article supplies that missing layer.

- **O5. [Why We Use Separate Tech Stacks for Personalization and Experimentation](https://engineering.atspotify.com/2026/01/why-we-use-separate-tech-stacks-for-personalization-and-experimentation)** — *Engineering article*; Spotify Engineering; January 2026. **System layer:** organizational and serving architecture. **Why read:** It clearly separates the infrastructure that serves low-latency personalized decisions from the infrastructure that establishes whether the whole recommender actually helps.

- **O6. [Bridging the Gap: Diagnosing Online-Offline Discrepancy in Pinterest’s L1 Conversion Models](https://medium.com/pinterest-engineering/bridging-the-gap-diagnosing-online-offline-discrepancy-in-pinterests-l1-conversion-models-1320faaaeefe)** — *Engineering article*; Pinterest Engineering; 2026. **System layer:** embedding versions, feature coverage, and funnel alignment. **Why read:** It shows how a model can improve offline while production quality stays flat because stale embeddings, missing features, or another funnel stage is the real bottleneck.

- **O7. [The little engine that could: Linchpin DSL for Pinterest ranking](https://medium.com/pinterest-engineering/the-little-engine-that-could-linchpin-dsl-for-pinterest-ranking-17699add8e56)** — *Engineering article*; Angela Sheu, Pinterest; 25 August 2017. **System layer:** model specification and consistent deployment. **Why read:** It highlights the less glamorous but essential problem of expressing, evaluating, and serving the same ranking computation across teams and environments.

### I. Experimentation, evaluation, safety, fairness, and ecosystem health

- **E1. [Beyond A/B Test: Speeding up Airbnb Search Ranking Experimentation through Interleaving](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e)** — *Engineering article*; Qing Zhang, Michelle Du, Reid Andersen, and Liwei He, Airbnb; 6 October 2022. **System layer:** online ranker evaluation. **Why read:** It connects NDCG, interleaving, attribution, and A/B tests while showing where interleaving breaks for set-level optimization.

- **E2. [A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation](https://arxiv.org/abs/2207.07483)** — *Paper*; Aleksandr Petrov and Craig Macdonald; 2022. **System layer:** offline evaluation and reproducibility. **Why read:** It demonstrates that training duration and implementation details can reverse benchmark conclusions, a warning to record exact data, negatives, seeds, and convergence conditions.

- **E3. [Towards Unified Metrics for Accuracy and Diversity for Recommender Systems](https://research.google/pubs/towards-unified-metrics-for-accuracy-and-diversity-for-recommender-systems/)** — *Paper*; Javier Parapar and Filip Radlinski, Google; RecSys 2021. **System layer:** offline list evaluation. **Why read:** It gives you a principled way to stop “more accurate” from meaning ten redundant variants of the same shoe.

- **E4. [Fairness in Recommendation Ranking through Pairwise Comparisons](https://research.google/pubs/fairness-in-recommendation-ranking-through-pairwise-comparisons/)** — *Paper*; Alex Beutel et al., Google; KDD 2019. **System layer:** ranking fairness metrics and training. **Why read:** It shows how randomized pairwise comparisons can expose and regularize ranking disparities instead of relying on aggregate click metrics.

- **E5. [Joint Multisided Exposure Fairness for Recommendation](https://research.google/pubs/joint-multisided-exposure-fairness-for-recommendation/)** — *Paper*; Haolun Wu, Bhaskar Mitra, Chen Ma, Fernando Diaz, and Xue Liu; SIGIR 2022. **System layer:** consumer and producer exposure. **Why read:** Myntra-like systems affect shoppers and sellers simultaneously, so you need to reason about who receives useful results and which products or brands receive opportunity.

- **E6. [Practical Compositional Fairness: Understanding Fairness in Multi-Component Recommender Systems](https://research.google/pubs/practical-compositional-fairness-understanding-fairness-in-multi-component-recommender-systems/)** — *Paper*; Xuezhi Wang et al., Google; WSDM 2021. **System layer:** end-to-end pipeline fairness. **Why read:** It warns that fair retrieval and fair ranking components do not automatically compose into a fair final shoe feed.

- **E7. [Feedback Loop and Bias Amplification in Recommender Systems](https://arxiv.org/abs/2007.13019)** — *Paper*; Masoud Mansoury, Himan Abdollahpouri, Mykola Pechenizkiy, Bamshad Mobasher, and Robin Burke; 2020. **System layer:** feedback dynamics and popularity bias. **Why read:** It gives you a simulation framework for checking whether early exposure makes popular shoes increasingly dominant and erases niche user tastes.

- **E8. [Modeling Recommender Ecosystems: Research Challenges at the Intersection of Mechanism Design, Reinforcement Learning and Generative Models](https://research.google/pubs/modeling-recommender-ecosystems-research-challenges-at-the-intersection-of-mechanism-design-reinforcement-learning-and-generative-models/)** — *Paper*; Craig Boutilier, Martin Mladenov, and Guy Tennenholtz, Google; AAAI 2024. **System layer:** marketplace ecosystem and long-horizon objectives. **Why read:** It expands the design target beyond one user’s next click to incentives, strategic sellers, long-term behavior, multiple stakeholders, and the role generative models might responsibly play.

## User-provided discovery sources and verification notes

- **Netflix GenRec:** direct article verified; included as G1 and in the must-read path.
- **LinkedIn Hiring Assistant semantic search:** direct article verified; included as R9/G2 and in the must-read path because it spans several layers.
- **Pinterest Engineering index:** homepage verified, but the homepage itself is not treated as a reading. Specific Pinterest articles are included throughout the catalog, notably Pixie, PinSage, learned retrieval, pre-ranking, sequence modeling, hybrid search, shopping conversion retrieval, freshness, serving, and multi-objective reranking.
- **[Tech with Mak X thread: “These engineering blogs have taught me more about AI system design than most courses... 15 worth bookmarking.”](https://x.com/techNmak/status/2091977224660656189)** — *Discovery index*; Tech with Mak; 25 August 2026. The parent URL, date, title, and broad-blog character were verified by Anshul in the signed-in browser. It is not a recommender-system article, so its 15 general engineering homepages were not promoted into this catalog; only directly relevant first-party RecSys sources were selected and verified separately.

## Suggested article thesis for Myntra Shoes

The reading list points toward a stronger thesis than “I added RAG to ecommerce”:

> I built a multimodal shoe-discovery system from scratch and tested where collaborative filtering, semantic retrieval, sequence models, ranking, and LLMs belong—and where catalog databases, hard filters, and ordinary search must remain authoritative.

The evidence for that article should compare at least four controlled systems on the same frozen query set:

1. metadata filters plus lexical search;
2. text embedding retrieval;
3. hybrid text, image, and behavioral retrieval plus a learned ranker;
4. the same pipeline with an LLM used only for intent parsing, labeling, reranking, or explanation.

Report per-query results, retrieval recall, ranking metrics, diversity, latency by stage, index freshness, contradictory-constraint behavior, cold-start behavior, and failure cases. Do not claim that an LLM improved recommendation unless it beats the non-LLM baseline under the same catalog, queries, labels, and evaluation protocol.
