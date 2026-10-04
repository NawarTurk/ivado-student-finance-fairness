# ÉquiAlgo: Fair Student Financing

Project for the Polytechnique Montréal AI hackathon challenge presented by IVADO. The challenge is to diagnose regional disparities in a synthetic scholarship and student-loan process, then produce fairer decisions for 4,000 new applicants.

## Challenge

Historical committee decisions grant scholarships to 48.4% of applicants from Montréal and Capitale-Nationale, compared with 27.3% from remote regions. Teams must submit predictions within a fixed 36–44% grant budget. Historical `decision_octroi` records committee choices; the evaluation applicants have no known outcomes, and scoring uses an independent hidden standard.

## Repository

- `equialgo-participants/`: challenge description and synthetic datasets (10,000 historical applications; 4,000 evaluation applicants).
- `analysis/fairness_and_bias_analysis/`: historical decision audits and baseline prediction analyses.
- `analysis/proxy_analysis/`: tests how application features reveal region.
- `models/`: logistic regression, random forest, and XGBoost experiments across feature sets and fairness settings.

The required submission is a root-level `predictions.csv` with one binary decision per evaluation applicant and a grant rate within the budget. Because evaluation outcomes are unknown, prediction audits report selection patterns, not accuracy against true awards.
