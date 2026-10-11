# Release decision: BLOCKED

**Candidate docqa 1.1.0** vs **production docqa 1.0.0**, decision `gate-20261011T034834-143043`.

| | Production | Candidate |
|---|---|---|
| Version | 1.0.0 | 1.1.0 |
| Model and prompt | qwen3.5:4b @ 3c3adf637187, prompt prompts/docqa/answer_v1.txt (eeb615b4) | qwen3.5:4b @ 3c3adf637187, prompt prompts/docqa/answer_v2.txt (4dba70ea) |
| Fingerprint | `c2b11ad8272d` | `7b48287c7a20` |
| Evaluation run | `docqa-v1.0.0-20261011T034830-5204` | `docqa-v1.1.0-20261011T034833-e73a` |
| Latency figures | recorded | recorded |

Dataset `policies/v1:test` (hash `a7fc1827c5e9`), 250 cases, scoring `996dae61ec36`, gate rules `be5865c1ce36`.

## Decision by dimension

| Dimension | Rule | Category (n) | Production | Candidate | Change [95% CI] | Requirement | Result |
|---|---|---|---|---|---|---|---|
| Answer quality | answer-correctness | answerable (210) | 84.0% | 88.3% | +4.3 pts [+1.9 pts, +6.9 pts] | >= 80.0%, drop <= 3.0 pts | **PASS** |
| Answer quality | answer-quality (advisory) | answerable (205) | 4.58 | 4.53 | -0.06 [-0.16, +0.04] | drop <= 0.30 | **PASS** |
| Abstention | no-false-abstentions | answerable (210) | 4.3% | 2.4% | -1.9 pts [-3.8 pts, -0.5 pts] | <= 8.0%, rise <= 3.0 pts | **PASS** |
| Abstention | declines-unanswerable | unanswerable (40) | 82.5% | 70.0% | -12.5 pts [-22.5 pts, -2.5 pts] | >= 75.0%, drop <= 7.5 pts | **FAIL** |
| Citations and grounding | citations-present | citation_required (205) | 0.0% | 0.0% | +0.0 pts [+0.0 pts, +0.0 pts] | <= 2.0% | **PASS** |
| Citations and grounding | citations-valid | citation_required (205) | 0.5% | 0.0% | -0.5 pts [-1.5 pts, +0.0 pts] | <= 2.0% | **PASS** |
| Citations and grounding | citation-support | citation_required (205) | 99.8% | 82.3% | -17.5 pts [-21.0 pts, -14.3 pts] | >= 95.0%, drop <= 3.0 pts | **FAIL** |
| Citations and grounding | unsupported-claims | all (217) | 0.2% | 9.8% | +9.6 pts [+7.4 pts, +12.1 pts] | <= 3.0%, rise <= 2.0 pts | **FAIL** |
| Citations and grounding | no-contradictions | all (217) | 0.0% | 0.0% | +0.0 pts [+0.0 pts, +0.0 pts] | <= 2.0% | **PASS** |
| Retrieval | retrieval-recall (advisory) | answerable (210) | 90.0% | 90.0% | +0.0 pts [+0.0 pts, +0.0 pts] | >= 85.0% | **PASS** |
| Application reliability | app-failures | all (250) | 0.0% | 0.0% | +0.0 pts [+0.0 pts, +0.0 pts] | <= 1.0% | **PASS** |
| Evaluation reliability | judge-unevaluable | all (217) | 0.0% | 0.0% | +0.0 pts [+0.0 pts, +0.0 pts] | <= 2.0% | **PASS** |
| Performance and cost | latency-p95 | all (250) | 2.34 s | 6.81 s | +4.47 s [+4.20 s, +4.81 s] | <= 4.00 s, rise <= 50% | **FAIL** |
| Performance and cost | cost (advisory) | all (250) | $0.0395 | $0.0571 | +$0.0176 [+$0.0161, +$0.0191] | rise <= 50% | **PASS** |

## Why

- **declines-unanswerable** (blocking, unanswerable): 70.0% is below the minimum 75.0%; dropped -12.5 pts vs production (allowed: -7.5 pts)
- **citation-support** (blocking, citation_required): 82.3% is below the minimum 95.0%; dropped -17.5 pts vs production (allowed: -3.0 pts)
- **unsupported-claims** (blocking, all): 9.8% is above the maximum 3.0%; rose +9.6 pts vs production (allowed: +2.0 pts)
- **latency-p95** (blocking, all): 6.81 s is above the maximum 4.00 s; rose 191% vs production (allowed: 50%)

## Results by category

| Category | Metric | Source | Production | Candidate | Change | 95% CI | Reading |
|---|---|---|---|---|---|---|---|
| answerable (210) | answer_correctness | judged | 84.0% | 88.3% | +4.3 pts | [+1.9 pts, +6.9 pts] | improved |
| answerable (210) | answer_quality | judged | 4.58 | 4.53 | -0.06 | [-0.16, +0.04] | no clear difference |
| answerable (210) | reference_f1 | deterministic | 49.8% | 16.2% | -33.6 pts | [-36.8 pts, -30.5 pts] | regressed |
| answerable (210) | false_abstention_rate | deterministic | 4.3% | 2.4% | -1.9 pts | [-3.8 pts, -0.5 pts] | improved |
| answerable (210) | retrieval_recall | deterministic | 90.0% | 90.0% | +0.0 pts | [+0.0 pts, +0.0 pts] | no clear difference |
| unanswerable (40) | correct_abstention_rate | deterministic | 82.5% | 70.0% | -12.5 pts | [-22.5 pts, -2.5 pts] | regressed |
| unanswerable (40) | inappropriate_answer_rate | deterministic | 17.5% | 30.0% | +12.5 pts | [+2.5 pts, +22.5 pts] | regressed |
| citation_required (210) | citation_missing_rate | deterministic | 0.0% | 0.0% | +0.0 pts | [+0.0 pts, +0.0 pts] | no clear difference |
| citation_required (210) | citation_invalid_rate | deterministic | 0.5% | 0.0% | -0.5 pts | [-1.5 pts, +0.0 pts] | no clear difference |
| citation_required (210) | citation_support_rate | judged | 99.8% | 82.3% | -17.5 pts | [-21.0 pts, -14.3 pts] | regressed |
| all (250) | abstention_accuracy | deterministic | 93.6% | 93.2% | -0.4 pts | [-2.8 pts, +2.0 pts] | no clear difference |
| all (250) | unsupported_claim_rate | judged | 0.2% | 9.8% | +9.6 pts | [+7.4 pts, +12.1 pts] | regressed |
| all (250) | contradiction_rate | judged | 0.0% | 0.0% | +0.0 pts | [+0.0 pts, +0.0 pts] | no clear difference |
| all (250) | app_failure_rate | deterministic | 0.0% | 0.0% | +0.0 pts | [+0.0 pts, +0.0 pts] | no clear difference |
| all (250) | judge_unevaluable_rate | deterministic | 0.0% | 0.0% | +0.0 pts | [+0.0 pts, +0.0 pts] | no clear difference |
| all (250) | latency_p50_s | measured | 1.65 s | 3.79 s | +2.14 s | [+1.95 s, +2.32 s] | regressed |
| all (250) | latency_p95_s | measured | 2.34 s | 6.81 s | +4.47 s | [+4.20 s, +4.81 s] | regressed |
| all (250) | est_cost_per_success_usd | estimated | $0.000050 | $0.000070 | +$0.000020 | [+$0.000017, +$0.000024] | regressed |

"No clear difference" means the 95% interval includes zero: on these cases the data cannot tell the versions apart on that metric, whatever the point estimates say.

## Cases behind the regressions

### declines-unanswerable (unanswerable): 5 cases got worse

| Case | Question | What changed | Production answer | Candidate answer |
|---|---|---|---|---|
| `enmfxdqz-q5` | How does real-time reporting enhance budget forecasting accuracy? | answered a question the document cannot answer | I could not find the answer to that in this document. | Real-time reporting enhances budget forecasting accuracy by providing immediate visibility into spending patterns, allowing management to adjust budgets proa... |
| `eqoorsnn-q5` | How does cultural diversity impact Glodev's expense policies? | answered a question the document cannot answer | I could not find the answer to that in this document. | Cultural diversity is deeply woven into the heart of Glodev Inc's expense policies, transforming what could be simple financial rules into opportunities for ... |
| `jujqwnlt-q4` | How has corporate travel affected Globex's international expansion strategy? | answered a question the document cannot answer | I could not find the answer to that in this document. | Corporate travel has been a driving force behind Globex's international expansion strategy. As commercial aviation connected the world in the mid-20th centur... |
| `nlzbyehu-q4` | What strategies will ensure inclusivity in a hyper-digital workspace? | answered a question the document cannot answer | I could not find the answer to that in this document. | To foster an inclusive environment in our hyper-digital workspace at Titan Dynamics, we focus on creating a respectful and accessible atmosphere where every ... |
| `odnkakms-q5` | Can remote work hinder spontaneous knowledge exchanges? | answered a question the document cannot answer | I could not find the answer to that in this document. | While it is true that remote work can sometimes make spontaneous, face-to-face interactions less frequent compared to an office environment, FutureLink Innov... |

### citation-support (citation_required): 90 cases got worse, showing 8

| Case | Question | What changed | Production answer | Candidate answer |
|---|---|---|---|---|
| `xatygnmj-q3` | What are some of the excluded items that BlueSky Innovations will not reimburse for remote work? | claims supported by its citations 2/2 -> 0/4 | Excluded items include but are not limited to those specified in Section 8, which states that BlueSky Innovations maintains clear boundaries regarding what i... | Great question! While we're happy to help cover many work-related costs, there are a few things that fall outside our reimbursement policy. Specifically, Blu... |
| `fquekjqs-q2` | Is multi-factor authentication required for Secureway Solutions employees when accessing company systems? | claims supported by its citations 1/1 -> 1/6 | Yes, multi-factor authentication is required for access to all company systems. | Yes, multi-factor authentication (MFA) is absolutely required for all Secureway Solutions employees when accessing company systems. This isn't just a rule on... |
| `fquekjqs-q3` | Are Secureway Solutions employees allowed to conduct company work on personal devices? | claims supported by its citations 1/1 -> 1/6 | No, personal devices must not be used for accessing Secureway Solutions' sensitive data unless expressly permitted and secure according to company standards. | Generally, no. Our policy states that all company work should be done on provided devices with up-to-date security software and encryption. However, there is... |
| `eagajflp-q4` | What is the typical timeframe for reimbursements after an expense report is approved by FintechReimburse Group? | claims supported by its citations 1/1 -> 1/4 | The typical timeframe for reimbursements is 10 days or less. | Great question! Once your expense report has been fully reviewed and approved by your manager, you can typically expect to receive reimbursement within 10 da... |
| `xatygnmj-q2` | What is the deadline for submitting an expense report after an expense is incurred at BlueSky Innovations? | claims supported by its citations 1/1 -> 1/4 | Employees must submit a detailed expense report no later than 30 days following the purchase. | At BlueSky Innovations, you have until **30 days** after you incur an expense to submit your detailed expense report. This ensures that all transactions are ... |
| `ybhxnrzb-q1` | What is the primary purpose of the International Travel Considerations Policy at Global Synergetics Inc.? | claims supported by its citations 2/2 -> 1/4 | The primary purpose of the policy is to ensure that employees conduct business in full compliance with the laws and regulations of host countries as well as ... | The primary purpose of the International Travel Considerations Policy at Global Synergetics Inc. is to ensure that employees conduct business while traveling... |
| `brdeefun-q1` | What is the main objective of the Employee Wellness Program at Solstice Innovations? | claims supported by its citations 1/1 -> 1/3 | The main objective is to ensure that employees maintain not just physical health, but also mental and emotional well-being. | The main objective of the Employee Wellness Program at Solstice Innovations is to ensure that employees maintain not just physical health, but also mental an... |
| `kcxllzsi-q2` | Does Zenith Innovations compensate for depreciation and wear and tear on employee vehicles used for business purposes? | claims supported by its citations 1/1 -> 2/6 | Yes, Zenith Innovations compensates for depreciation and wear and tear on employee vehicles used for business purposes. | Yes, absolutely! According to our Travel Policy, we do compensate employees for the depreciation and wear and tear on their personal vehicles when they are u... |

### unsupported-claims (all): 57 cases got worse, showing 8

| Case | Question | What changed | Production answer | Candidate answer |
|---|---|---|---|---|
| `fquekjqs-q2` | Is multi-factor authentication required for Secureway Solutions employees when accessing company systems? | unsupported claims 0 -> 5: "MFA is a key part of keeping data and the business safe from unauthorized access." | Yes, multi-factor authentication is required for access to all company systems. | Yes, multi-factor authentication (MFA) is absolutely required for all Secureway Solutions employees when accessing company systems. This isn't just a rule on... |
| `ybhxnrzb-q1` | What is the primary purpose of the International Travel Considerations Policy at Global Synergetics Inc.? | unsupported claims 0 -> 3: "The policy helps employees navigate complex international environments by reminding them to respect local customs, protect sensit... | The primary purpose of the policy is to ensure that employees conduct business in full compliance with the laws and regulations of host countries as well as ... | The primary purpose of the International Travel Considerations Policy at Global Synergetics Inc. is to ensure that employees conduct business while traveling... |
| `fquekjqs-q3` | Are Secureway Solutions employees allowed to conduct company work on personal devices? | unsupported claims 0 -> 3: "A personal device can be used if it has been expressly permitted by a manager and meets specific security standards." | No, personal devices must not be used for accessing Secureway Solutions' sensitive data unless expressly permitted and secure according to company standards. | Generally, no. Our policy states that all company work should be done on provided devices with up-to-date security software and encryption. However, there is... |
| `gkzgogqd-q1` | What is the metaphorical phrase used to describe the beginning of Endless Frontier, Inc.'s journey in their welcome message? | unsupported claims 0 -> 1: "This line humorously sets the tone for the company's approach to professional development, suggesting that even the start of thei... | A journey of a thousand webinars begins with a single unstable Internet connection. | The metaphorical phrase used to describe the beginning of the journey in the welcome message is "A journey of a thousand webinars begins with a single unstab... |
| `gkzgogqd-q4` | What is the availability of a mentor in the Endless Frontier, Inc.’s Mentorship Program? | unsupported claims 0 -> 1: "Since there typically aren't five Sundays in a month, you might need to keep checking the calendar or perhaps send a message earl... | The assigned mentor is reachable exclusively through carrier pigeon between the hours of 12:01 am and 3:00 am on the fifth Sunday of every month. | The Endless Frontier, Inc. Mentorship Program offers a very unique way to connect with your assigned 'Yoda' figure! Your mentor will be reachable exclusively... |
| `jujqwnlt-q1` | What is the importance of technology incorporation in Globex's corporate travel policy? | unsupported claims 0 -> 3: "These tools help employees stay organized on the go without carrying heavy paperwork or getting bogged down by manual processes." | Technology incorporation has streamlined the process of international travel, making it more manageable and less time-consuming by providing tools such as mo... | Technology plays a huge role in making your travels smoother and less stressful at Globex! Thanks to our digital revolution, you now have access to powerful ... |
| `sbjzpntk-q1` | What is the purpose of the conversations over lunch at Growth Horizons Inc.? | unsupported claims 0 -> 2: "They are a vibrant community hub where employees celebrate and support each other's ambitions." | The purpose of the conversations over lunch is to discuss courses taken as part of the Tuition Reimbursement Program and share experiences in pursuing degree... | At Growth Horizons Inc., our lunchtime conversations are more than just a chance to refuel; they are a vibrant community hub where we celebrate and support e... |
| `vglnhovm-q3` | What strategies does AlliedTech Industries employ to promote allyship within diverse teams? | unsupported claims 0 -> 3: "Allyship should be viewed as an ongoing practice rather than a one-time task." | AlliedTech Industries promotes allyship by encouraging team members to educate themselves on diversity issues, stand against injustice, create opportunities ... | At AlliedTech Industries, promoting allyship is about moving beyond simple support to become actively engaged advocates for your colleagues. We believe that ... |

### latency-p95 (all): 216 cases got worse, showing 8

| Case | Question | What changed | Production answer | Candidate answer |
|---|---|---|---|---|
| `odnkakms-q5` | Can remote work hinder spontaneous knowledge exchanges? | original latency 1.07 s -> 8.14 s | I could not find the answer to that in this document. | While it is true that remote work can sometimes make spontaneous, face-to-face interactions less frequent compared to an office environment, FutureLink Innov... |
| `fquekjqs-q2` | Is multi-factor authentication required for Secureway Solutions employees when accessing company systems? | original latency 1.19 s -> 7.62 s | Yes, multi-factor authentication is required for access to all company systems. | Yes, multi-factor authentication (MFA) is absolutely required for all Secureway Solutions employees when accessing company systems. This isn't just a rule on... |
| `duthzmzk-q5` | What method does GlobalFinance Corp. use to reimburse approved expenses to its employees? | original latency 1.80 s -> 7.89 s | GlobalFinance Corp. reimburses approved expenses through direct deposit, typically appearing in employee bank accounts within three business days after appro... | Great question! Once your expense report is submitted and approved, GlobalFinance Corp. handles reimbursement through direct deposit into your bank account. ... |
| `dvxzgslm-q2` | What purpose do the Employee Resource Groups (ERGs) serve at Harmony United? | original latency 2.37 s -> 8.28 s | The Employee Resource Groups (ERGs) serve as a platform for personal and professional growth through shared experiences, support, and advocacy, acting as int... | At Harmony United, our Employee Resource Groups (ERGs) are designed to be more than just formal committees; they function as intimate families and vibrant co... |
| `eqoorsnn-q5` | How does cultural diversity impact Glodev's expense policies? | original latency 1.06 s -> 6.91 s | I could not find the answer to that in this document. | Cultural diversity is deeply woven into the heart of Glodev Inc's expense policies, transforming what could be simple financial rules into opportunities for ... |
| `tetifeio-q2` | What is the expectation for Nexus Innovations Inc. employees' availability during business hours? | original latency 1.38 s -> 7.07 s | Employees must be available during core business hours as defined by their team's time zone. | At Nexus Innovations Inc., we believe that clear communication and collaboration are key to our success, no matter where you're working from! To ensure every... |
| `luxtaiqw-q3` | How does Verdant Visions Incorporated ensure the practical implementation of their anti-discrimination and anti-harassment policies? | original latency 1.97 s -> 7.63 s | Verdant Visions Incorporated ensures practical implementation through training sessions, a clear reporting structure, and protocols for prompt and thorough i... | At Verdant Visions Incorporated, we believe that having a policy is only the first step; making it work in real life requires everyone's active participation... |
| `sbjzpntk-q2` | How does Growth Horizons Inc. support employees with furthering their education during working hours? | original latency 1.05 s -> 6.62 s | I could not find the answer to that in this document. | At Growth Horizons Inc., we are deeply committed to supporting your personal and professional growth, including pursuing further education while you're here.... |


Thresholds apply to point estimates, the conservative choice for blocking regressions. The decision is stored with every rule result and the versions above, so it can be audited and reproduced. `evalops gate` never changes production; `evalops promote` only accepts a passing decision whose versions still match.
