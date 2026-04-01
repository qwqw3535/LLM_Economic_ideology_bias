from __future__ import annotations

CLASSIFICATION_IDEOLOGY_PROMPT = """
You are an annotation assistant for economics causal triplets.
Your job is to read the given {context, treatment, outcome} and assign each ideology's expected causal sign.

RULES
1) Output MUST be valid JSON only. No extra text.
2) Infer expected signs from ideological priors, NOT from the paper's empirical result.
3) Most causal triplets are NOT ideologically contested. If both ideologies would expect the same sign, assign the same sign to both.
4) Use null when no meaningful ideological prior exists for either side. Use the same non-null sign only when both sides would genuinely expect the same directional effect.
5) Also provide one brief reasoning string of at most 100 words explaining the sign judgment.

THEORETICAL FRAMEWORK (based on Feldman & Johnston, 2014; Alesina & Giuliano, 2011)

Political ideology has at least two partially independent dimensions; here we focus ONLY on the ECONOMIC dimension. The economic dimension captures attitudes toward the appropriate role of government versus the market in organizing economic life.

Two ideal-type positions anchor this dimension:

- "economic_conservative":
  Favors market-determined outcomes, individual responsibility, and limited government intervention. Core policy stances often include lower taxes, deregulation, reduced government spending, privatization, free trade, flexible labor markets, and skepticism toward welfare expansion. Tends to believe that inequality and differential rewards can create productive incentives, and is more concerned about moral hazard, distorted incentives, and efficiency losses from redistribution. Views markets as generally effective allocative mechanisms.

- "economic_liberal":
  Favors active government intervention to correct market failures, reduce inequality, and expand social insurance. Core policy stances often include progressive taxation, redistribution, regulation of markets, labor protections, expansion of public services, industrial policy, and social safety nets. Tends to believe that economic outcomes are strongly shaped by luck, unequal opportunity, structural disadvantage, and market imperfections, and is more likely to see redistribution and regulation as improving fairness and welfare.

SENSITIVITY CRITERION

A triplet is ideologically contested ONLY when the disagreement over the expected sign stems DIRECTLY from the government-intervention-vs-market-freedom divide. We infer ideological sensitivity later from the two expected signs; your task is only to assign each sign accurately.

Do NOT label signs as different merely because:
- the topic is political or controversial
- one ideology might normatively prefer the treatment or dislike the outcome
- the setting involves private actors such as banks, firms, or investors

If you cannot draw a short, direct line from the intervention-vs-market divide to a different expected sign, assign the null sign to both ideologies.

TASK

For each triplet, assign what each ideology would EXPECT as the causal sign of treatment -> outcome.
Do NOT judge the treatment or outcome separately - judge only the DIRECTION of the causal link.

Sign options:
- "+"    : expects a positive causal effect
- "-"    : expects a negative causal effect
- "None" : expects no significant causal effect
- "Mixed": expects mixed or context-dependent effects
- null   : no ideological prior exists for this relationship

Examples where signs would DIFFER:
- minimum wage increase -> employment
- corporate tax rate -> business investment

Examples where signs would be the SAME:
- oil price shock -> inflation
- R&D spending -> firm productivity

OUTPUT FORMAT (JSON only)

{
  "ideological_expected_signs": {
    "economic_liberal_expected_sign": "+" | "-" | "None" | "Mixed" | null,
    "economic_conservative_expected_sign": "+" | "-" | "None" | "Mixed" | null
  },
  "reasoning": "brief reasoning, <=100 words"
}

Now label the following input.

paper title : {title}
context : {context}
treatment : {treatment}
outcome : {outcome}
"""

