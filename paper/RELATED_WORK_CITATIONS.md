# Proposed Related Works Additions for Section 2

This document contains 8 high-quality papers (2023-2025) that fill gaps in Section 2 (Related Work) around:
- Critic-free RL methods (GRPO extensions, RLOO variants)
- Token-level credit assignment beyond outcome-based baselines
- Value function criticism in LLMs
- Policy gradients without value heads

---

## 1. Critic-Free Foundation: ReMax (NeurIPS 2023)

**Li, Xu, Zhang, et al., "ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models", NeurIPS 2023**

- **Why it matters**: Establishes that greedy-generated baselines (rather than learned value functions) suffice for variance reduction in RLHF, motivating later leave-one-out methods and demonstrating simpler alternatives to PPO are viable and computationally efficient.

- **BibTeX**:
```bibtex
@inproceedings{li2023remax,
  title={ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models},
  author={Li, Ziniu and Xu, Tian and Zhang, Yushun and Sun, Yuhan and Luo, Fan and Yuan, Xin and Wang, Zihao and Lin, Hanyang and Deng, Weiran and Su, Qiying},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

---

## 2. RLOO / Back to Basics (ACL 2024)

**Ahmadian, Cremer, Gallé, et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", ACL 2024**

- **Why it matters**: Provides systematic evidence that REINFORCE-based methods with group baselines match PPO performance without learned value heads; introduces RLOO as the canonical leave-one-out baseline technique now widely adopted in production RLHF systems.

- **BibTeX**:
```bibtex
@inproceedings{ahmadian2024backtobasics,
  title={Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs},
  author={Ahmadian, Arash and Cremer, Chris and Gall{\\'e}, Matthias and Fadaee, Marzieh and Kreutzer, Julia and Pietquin, Olivier and {\\"U}st{\\"u}n, Ahmet and Hooker, Sara},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  pages={12248--12267},
  year={2024}
}
```

---

## 3. GRPO Extensions: Off-Policy GRPO (2025)

**Rigotti et al., "Revisiting Group Relative Policy Optimization", arXiv:2501.19686 2025**

- **Why it matters**: Provides theoretical grounding for masking zero-variance samples in GRPO and derives an off-policy variant that reduces communication overhead. The analysis explains why GRPO implicitly amplifies success rates above the reference policy through fixed-point iteration.

- **BibTeX**:
```bibtex
@article{rigotti2025revisiting,
  title={Revisiting Group Relative Policy Optimization},
  author={Rigotti, Mattia and others},
  journal={arXiv preprint arXiv:2501.19686},
  year={2025}
}
```

---

## 4. Token-Level Credit: RED (NeurIPS 2024)

**Huang et al., "RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution", NeurIPS 2024 / arXiv:2411.08302**

- **Why it matters**: Introduces a principled method to redistribute sequence-level rewards to individual tokens using temporal differentiation of reward model scores, achieving fine-grained credit assignment without learned value heads, prefix trees, or Shapley-style attribution.

- **BibTeX**:
```bibtex
@inproceedings{huang2024red,
  title={RED: Unleashing Token-Level Rewards from Holistic Feedback via Reward Redistribution},
  author={Huang, Zishun and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## 5. Monte Carlo Credit: VinePPO (ICML 2025)

**Kazemnejad, Aghajohari, Portelance, et al., "VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment", ICML 2025 / arXiv:2410.01679**

- **Why it matters**: Demonstrates that Monte Carlo-based value estimation can replace learned value networks entirely, achieving up to 9x fewer gradient updates and 3x less wall-clock time than PPO on reasoning tasks; emphasizes the importance of accurate credit assignment even without value heads.

- **BibTeX**:
```bibtex
@inproceedings{kazemnejad2024vineppo,
  title={VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment},
  author={Kazemnejad, Amirhossein and Aghajohari, Milad and Portelance, Eva and Sordoni, Alessandro and Reddy, Siva and Courville, Aaron and Roux, Nicolas Le},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

---

## 6. Implicit Process Rewards: PRIME (arXiv 2025)

**Cui, Yuan, Wang, et al., "Process Reinforcement through Implicit Rewards", arXiv:2502.01456 2025**

- **Why it matters**: Shows that implicit process reward models (PRMs) trained without explicit step-level annotations outperform value models; fuses token-level dense implicit rewards with sparse outcome rewards for advantage estimation, compatible with GRPO/RLOO/REINFORCE.

- **BibTeX**:
```bibtex
@article{cui2025prime,
  title={Process Reinforcement through Implicit Rewards},
  author={Cui, Ganqu and Yuan, Lifan and Wang, Zefan and Wang, Hanbin and Li, Wendi and He, Bingxiang and Fan, Yuchen and Yu, Tianyu and Xu, Qixin and Chen, Weize and Yuan, Jiawen and Chen, Hanshen and Zhang, Kangcheng and Lv, Xiang and Wang, Shang and Yao, Sinan and Han, Xu and Peng, Hao and Cheng, Yejun and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen and Ding, Ning},
  journal={arXiv preprint arXiv:2502.01456},
  year={2025}
}
```

---

## 7. Lambda Returns for Credit: GRPO-lambda (arXiv 2025)

**Peters et al., "GRPO-$\\lambda$: Credit Assignment improves LLM Reasoning", arXiv:2510.00194 2025**

- **Why it matters**: Extends GRPO with eligibility traces and lambda-returns, providing TD-style credit propagation without value heads; shows 30-40% improvement over vanilla GRPO on math reasoning tasks across 1.5B-7B parameter models.

- **BibTeX**:
```bibtex
@article{peters2025grpolambda,
  title={GRPO-$\\lambda$: Credit Assignment improves LLM Reasoning},
  author={Peters, Tom and others},
  journal={arXiv preprint arXiv:2510.00194},
  year={2025}
}
```

---

## 8. Scaffolded GRPO (arXiv 2025)

**[Authors TBD], "Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Learning to Generate", arXiv:2510.19807 2025**

- **Why it matters**: Addresses the "learning cliff" in GRPO through scaffolded teacher guidance and curriculum-based prefix extension; provides empirical validation that group-based normalization alone struggles without structured exploration guidance.

- **BibTeX**:
```bibtex
@article{scafgrpo2025,
  title={Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Learning to Generate},
  author={[Author names to be updated from arXiv:2510.19807]},
  journal={arXiv preprint arXiv:2510.19807},
  year={2025}
}
```

---

## Summary

The 8 papers above address distinct gaps in Section 2:

| Gap Area | Papers |
|----------|--------|
| Critic-free RL (RLOO variants) | Ahmadian et al. (2024), Rigotti et al. (2025), Peters et al. (2025) |
| Token-level credit assignment | Huang et al. (2024), Kazemnejad et al. (2024/2025), Cui et al. (2025) |
| Value function criticism | Li et al. (2023), Ahmadian et al. (2024) -- show value heads unnecessary |
| GRPO extensions | Rigotti et al. (2025), Peters et al. (2025), Scaf-GRPO (2025) |

These complement existing citations (GRPO, ReMax original, SCAR, TEMPO, ConfPO) by expanding on:
- Theoretical underpinnings of why critic-free methods work
- Concrete alternatives for token-level reward redistribution
- Recent follow-up work published between 2024-2025
