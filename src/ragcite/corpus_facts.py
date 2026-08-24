"""Single source of truth for the synthetic regulated corpus.

Each fact becomes one paragraph in a generated source document *and* one
question in the golden evaluation set, so every golden answer is
guaranteed to be grounded in a specific, locatable passage. This keeps the
demo corpus small while making citation-grounding and RAGAS-style metrics
meaningful rather than hand-waved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    doc_id: str
    section: str
    statement: str
    question: str
    answer: str
    keywords: tuple[str, ...]


DOCS = {
    "clinical_guideline_diabetes": {
        "title": "Clinical Practice Guideline: Management of Type 2 Diabetes Mellitus",
        "filename": "clinical_guideline_diabetes.pdf",
        "kind": "pdf",
        "domain": "Life Sciences / Clinical",
    },
    "financial_reg_capital_adequacy": {
        "title": "Regulation FR-14: Capital Adequacy and Liquidity Requirements for Depository Institutions",
        "filename": "financial_reg_capital_adequacy.pdf",
        "kind": "pdf",
        "domain": "Financial Regulation",
    },
    "data_privacy_statute": {
        "title": "Data Privacy Compliance Statute, Sections 101-140",
        "filename": "data_privacy_statute.pdf",
        "kind": "pdf",
        "domain": "Legal / Statute",
    },
    "insurance_policy_group402": {
        "title": "Comprehensive Health Insurance Policy Document, Group Plan 402",
        "filename": "insurance_policy_group402.docx",
        "kind": "docx",
        "domain": "Insurance Policy",
    },
    "lab_sop_document_control": {
        "title": "Standard Operating Procedure SOP-QA-011: Document Control and Change Management",
        "filename": "lab_sop_document_control.docx",
        "kind": "docx",
        "domain": "Life Sciences / Quality SOP",
    },
}

FACTS: list[Fact] = [
    # ---- clinical_guideline_diabetes ----
    Fact(
        "diab-01", "clinical_guideline_diabetes", "Screening and Diagnosis",
        "Adults aged 35 years or older without symptoms of hyperglycemia should be screened for type 2 diabetes at least every 3 years using fasting plasma glucose, HbA1c, or a 75-gram oral glucose tolerance test.",
        "At what age should asymptomatic adults begin routine screening for type 2 diabetes, and how often?",
        "Adults aged 35 or older should be screened at least every 3 years.",
        ("35", "screen", "3 years"),
    ),
    Fact(
        "diab-02", "clinical_guideline_diabetes", "Screening and Diagnosis",
        "A diagnosis of diabetes is confirmed by an HbA1c of 6.5% or greater, a fasting plasma glucose of 126 mg/dL or greater, or a 2-hour plasma glucose of 200 mg/dL or greater during an oral glucose tolerance test.",
        "Which HbA1c threshold confirms a diagnosis of diabetes?",
        "An HbA1c of 6.5% or greater confirms a diagnosis of diabetes.",
        ("HbA1c", "6.5%"),
    ),
    Fact(
        "diab-03", "clinical_guideline_diabetes", "Screening and Diagnosis",
        "Individuals with a body mass index of 25 kg/m2 or higher who have one additional risk factor, such as a first-degree relative with diabetes or a history of gestational diabetes, should be screened at any age.",
        "At what BMI should adults with an additional risk factor be screened for diabetes regardless of age?",
        "Adults with a BMI of 25 kg/m2 or higher and at least one additional risk factor should be screened at any age.",
        ("BMI", "25", "risk factor"),
    ),
    Fact(
        "diab-04", "clinical_guideline_diabetes", "Pharmacologic Therapy",
        "Metformin remains the preferred initial pharmacologic agent for most patients with type 2 diabetes, started at diagnosis alongside lifestyle intervention unless contraindicated by an estimated glomerular filtration rate below 30 mL/min/1.73m2.",
        "What is the preferred initial pharmacologic agent for most patients with type 2 diabetes?",
        "Metformin is the preferred initial pharmacologic agent, started at diagnosis unless eGFR is below 30 mL/min/1.73m2.",
        ("metformin", "initial", "30 mL/min"),
    ),
    Fact(
        "diab-05", "clinical_guideline_diabetes", "Pharmacologic Therapy",
        "For patients with established atherosclerotic cardiovascular disease, a glucagon-like peptide-1 receptor agonist or a sodium-glucose cotransporter-2 inhibitor with demonstrated cardiovascular benefit should be added independent of HbA1c or metformin use.",
        "Which drug classes should be added for patients with established atherosclerotic cardiovascular disease, independent of HbA1c?",
        "A GLP-1 receptor agonist or an SGLT2 inhibitor with demonstrated cardiovascular benefit should be added.",
        ("GLP-1", "SGLT2", "cardiovascular"),
    ),
    Fact(
        "diab-06", "clinical_guideline_diabetes", "Pharmacologic Therapy",
        "The general HbA1c target for most nonpregnant adults with diabetes is below 7.0%, though a less stringent target of below 8.0% may be appropriate for patients with limited life expectancy or extensive comorbidity.",
        "What is the general HbA1c target for most nonpregnant adults with diabetes?",
        "The general HbA1c target for most nonpregnant adults is below 7.0%.",
        ("HbA1c", "7.0%", "target"),
    ),
    Fact(
        "diab-07", "clinical_guideline_diabetes", "Monitoring and Complications",
        "Comprehensive foot examinations, including assessment of pulses, sensation, and skin integrity, should be performed at least annually for all patients with diabetes.",
        "How often should a comprehensive foot examination be performed for patients with diabetes?",
        "A comprehensive foot examination should be performed at least annually.",
        ("foot examination", "annually"),
    ),
    Fact(
        "diab-08", "clinical_guideline_diabetes", "Monitoring and Complications",
        "Urine albumin-to-creatinine ratio and estimated glomerular filtration rate should be assessed at least once per year in all patients with type 2 diabetes to screen for diabetic kidney disease.",
        "Which two tests screen for diabetic kidney disease and how frequently should they be performed?",
        "Urine albumin-to-creatinine ratio and estimated GFR should be assessed at least once per year.",
        ("albumin-to-creatinine", "GFR", "once per year"),
    ),
    Fact(
        "diab-09", "clinical_guideline_diabetes", "Monitoring and Complications",
        "A dilated eye examination for diabetic retinopathy should occur within five years of diagnosis for type 1 diabetes and at the time of diagnosis for type 2 diabetes, then at least every two years thereafter if no retinopathy is found.",
        "When should the first dilated eye examination occur for a patient newly diagnosed with type 2 diabetes?",
        "The first dilated eye examination should occur at the time of diagnosis for type 2 diabetes.",
        ("dilated eye examination", "time of diagnosis"),
    ),
    Fact(
        "diab-10", "clinical_guideline_diabetes", "Special Populations",
        "In pregnant patients with pre-existing diabetes, the target HbA1c prior to conception is below 6.5% when it can be achieved without significant hypoglycemia.",
        "What is the pre-conception HbA1c target for pregnant patients with pre-existing diabetes?",
        "The pre-conception HbA1c target is below 6.5%, when achievable without significant hypoglycemia.",
        ("pregnant", "6.5%", "pre-conception"),
    ),
    Fact(
        "diab-11", "clinical_guideline_diabetes", "Special Populations",
        "Screening for gestational diabetes should be performed at 24 to 28 weeks of gestation using either a one-step 75-gram oral glucose tolerance test or a two-step approach with a 50-gram screen followed by a 100-gram confirmatory test.",
        "During which weeks of gestation should screening for gestational diabetes be performed?",
        "Screening for gestational diabetes should be performed at 24 to 28 weeks of gestation.",
        ("gestational", "24", "28 weeks"),
    ),
    Fact(
        "diab-12", "clinical_guideline_diabetes", "Special Populations",
        "In older adults with diabetes and significant comorbidity or cognitive impairment, an HbA1c target of below 8.0% to 8.5% is reasonable to reduce the risk of hypoglycemia.",
        "What HbA1c target range is reasonable for older adults with significant comorbidity or cognitive impairment?",
        "An HbA1c target of below 8.0% to 8.5% is reasonable for these patients.",
        ("older adults", "8.0%", "8.5%"),
    ),
    Fact(
        "diab-13", "clinical_guideline_diabetes", "Lifestyle Management",
        "Patients with type 2 diabetes should aim for at least 150 minutes per week of moderate-to-vigorous aerobic physical activity, spread over at least 3 days per week with no more than 2 consecutive days without activity.",
        "How many minutes per week of moderate-to-vigorous aerobic activity are recommended for patients with type 2 diabetes?",
        "At least 150 minutes per week, spread over at least 3 days.",
        ("150 minutes", "aerobic", "3 days"),
    ),
    Fact(
        "diab-14", "clinical_guideline_diabetes", "Lifestyle Management",
        "Medical nutrition therapy delivered by a registered dietitian should be offered to all patients with diabetes at diagnosis and reassessed at least annually.",
        "Who should deliver medical nutrition therapy to patients with diabetes, and how often should it be reassessed?",
        "A registered dietitian should deliver it, reassessed at least annually.",
        ("registered dietitian", "annually", "medical nutrition therapy"),
    ),
    Fact(
        "diab-15", "clinical_guideline_diabetes", "Blood Pressure and Lipid Management",
        "Blood pressure should be measured at every routine clinical visit, with a general target of below 130/80 mmHg for most patients with diabetes and hypertension.",
        "What is the general blood pressure target for most patients with diabetes and hypertension?",
        "The general target is below 130/80 mmHg.",
        ("blood pressure", "130/80"),
    ),
    Fact(
        "diab-16", "clinical_guideline_diabetes", "Blood Pressure and Lipid Management",
        "A moderate-intensity or high-intensity statin should be prescribed for all patients with diabetes aged 40 to 75 years, in addition to lifestyle therapy, regardless of baseline lipid levels.",
        "For which age range should a statin be prescribed to patients with diabetes regardless of baseline lipid levels?",
        "A statin should be prescribed for patients aged 40 to 75 years.",
        ("statin", "40", "75 years"),
    ),

    # ---- financial_reg_capital_adequacy ----
    Fact(
        "fin-01", "financial_reg_capital_adequacy", "Minimum Capital Ratios",
        "A covered depository institution must maintain a Common Equity Tier 1 capital ratio of at least 4.5 percent of total risk-weighted assets at all times.",
        "What is the minimum Common Equity Tier 1 capital ratio a covered depository institution must maintain?",
        "A minimum Common Equity Tier 1 capital ratio of 4.5 percent of risk-weighted assets.",
        ("Common Equity Tier 1", "4.5 percent"),
    ),
    Fact(
        "fin-02", "financial_reg_capital_adequacy", "Minimum Capital Ratios",
        "The minimum total capital ratio, comprising Tier 1 and Tier 2 capital, is set at 8.0 percent of total risk-weighted assets.",
        "What is the minimum total capital ratio required under this regulation?",
        "The minimum total capital ratio is 8.0 percent of risk-weighted assets.",
        ("total capital ratio", "8.0 percent"),
    ),
    Fact(
        "fin-03", "financial_reg_capital_adequacy", "Minimum Capital Ratios",
        "In addition to minimum ratios, a capital conservation buffer of 2.5 percent of risk-weighted assets, composed entirely of Common Equity Tier 1 capital, must be maintained to avoid restrictions on capital distributions.",
        "What is the required capital conservation buffer, and what type of capital must compose it?",
        "A 2.5 percent capital conservation buffer composed entirely of Common Equity Tier 1 capital.",
        ("capital conservation buffer", "2.5 percent"),
    ),
    Fact(
        "fin-04", "financial_reg_capital_adequacy", "Minimum Capital Ratios",
        "An institution designated as a global systemically important bank must hold an additional common equity surcharge ranging from 1.0 to 3.5 percent of risk-weighted assets, determined by its systemic risk score.",
        "What additional capital surcharge applies to a global systemically important bank?",
        "An additional common equity surcharge of 1.0 to 3.5 percent of risk-weighted assets.",
        ("systemically important", "surcharge", "1.0", "3.5 percent"),
    ),
    Fact(
        "fin-05", "financial_reg_capital_adequacy", "Liquidity Coverage Ratio",
        "Covered institutions must maintain a Liquidity Coverage Ratio of at least 100 percent, ensuring high-quality liquid assets are sufficient to cover total net cash outflows over a 30-calendar-day stress period.",
        "What is the minimum Liquidity Coverage Ratio and over what stress period is it measured?",
        "A minimum LCR of 100 percent over a 30-calendar-day stress period.",
        ("Liquidity Coverage Ratio", "100 percent", "30-calendar-day"),
    ),
    Fact(
        "fin-06", "financial_reg_capital_adequacy", "Liquidity Coverage Ratio",
        "High-quality liquid assets are classified into Level 1 assets, which are not subject to a haircut, and Level 2 assets, which are subject to haircuts of 15 to 50 percent depending on subcategory.",
        "What haircut range applies to Level 2 high-quality liquid assets?",
        "Level 2 assets are subject to haircuts of 15 to 50 percent depending on subcategory.",
        ("Level 2", "haircut", "15", "50 percent"),
    ),
    Fact(
        "fin-07", "financial_reg_capital_adequacy", "Liquidity Coverage Ratio",
        "Level 2 assets in aggregate may not exceed 40 percent of an institution's total stock of high-quality liquid assets after applicable haircuts.",
        "What is the maximum share of total high-quality liquid assets that Level 2 assets may represent?",
        "Level 2 assets may not exceed 40 percent of total high-quality liquid assets.",
        ("Level 2", "40 percent"),
    ),
    Fact(
        "fin-08", "financial_reg_capital_adequacy", "Stress Testing",
        "Covered institutions with total consolidated assets of 100 billion dollars or more must conduct an annual company-run stress test under baseline, adverse, and severely adverse scenarios published by the regulator.",
        "Which institutions are required to conduct an annual company-run stress test, and under how many scenarios?",
        "Institutions with total consolidated assets of 100 billion dollars or more, under three scenarios: baseline, adverse, and severely adverse.",
        ("100 billion", "stress test", "severely adverse"),
    ),
    Fact(
        "fin-09", "financial_reg_capital_adequacy", "Stress Testing",
        "Results of the annual company-run stress test must be submitted to the regulator no later than April 5 of each calendar year and published in a public summary within 15 days of submission.",
        "By what date must annual stress test results be submitted to the regulator?",
        "Results must be submitted no later than April 5 of each calendar year.",
        ("April 5", "stress test", "submitted"),
    ),
    Fact(
        "fin-10", "financial_reg_capital_adequacy", "Stress Testing",
        "An institution that fails to maintain a post-stress Common Equity Tier 1 ratio above 4.5 percent under the severely adverse scenario must submit a capital remediation plan within 30 days.",
        "What post-stress Common Equity Tier 1 ratio threshold triggers a required capital remediation plan?",
        "A post-stress CET1 ratio at or below 4.5 percent under the severely adverse scenario triggers a remediation plan, due within 30 days.",
        ("post-stress", "4.5 percent", "remediation plan", "30 days"),
    ),
    Fact(
        "fin-11", "financial_reg_capital_adequacy", "Reporting Requirements",
        "Covered institutions must file the quarterly regulatory capital report, Form FR-14Q, no later than 30 calendar days after the end of each fiscal quarter.",
        "What is the filing deadline for the quarterly regulatory capital report, Form FR-14Q?",
        "Form FR-14Q must be filed no later than 30 calendar days after the end of each fiscal quarter.",
        ("FR-14Q", "30 calendar days", "quarterly"),
    ),
    Fact(
        "fin-12", "financial_reg_capital_adequacy", "Reporting Requirements",
        "Material errors discovered in a previously filed capital report must be disclosed to the regulator within 5 business days of discovery, along with a corrected filing.",
        "Within how many business days must a material error in a previously filed capital report be disclosed?",
        "Material errors must be disclosed within 5 business days of discovery.",
        ("material error", "5 business days"),
    ),
    Fact(
        "fin-13", "financial_reg_capital_adequacy", "Reporting Requirements",
        "The chief financial officer or equivalent senior officer must personally attest to the accuracy of each quarterly capital report prior to submission.",
        "Who must personally attest to the accuracy of each quarterly capital report?",
        "The chief financial officer or an equivalent senior officer must attest to its accuracy.",
        ("chief financial officer", "attest"),
    ),
    Fact(
        "fin-14", "financial_reg_capital_adequacy", "Net Stable Funding Ratio",
        "Covered institutions must maintain a Net Stable Funding Ratio of at least 100 percent, comparing available stable funding to required stable funding over a one-year horizon.",
        "What is the minimum Net Stable Funding Ratio required, and over what time horizon is it calculated?",
        "A minimum NSFR of 100 percent over a one-year horizon.",
        ("Net Stable Funding Ratio", "100 percent", "one-year"),
    ),
    Fact(
        "fin-15", "financial_reg_capital_adequacy", "Enforcement and Penalties",
        "An institution that fails to meet minimum capital ratios for two consecutive quarters is classified as undercapitalized and becomes subject to mandatory restrictions on asset growth and executive compensation.",
        "What happens to an institution that fails to meet minimum capital ratios for two consecutive quarters?",
        "It is classified as undercapitalized and becomes subject to restrictions on asset growth and executive compensation.",
        ("undercapitalized", "two consecutive quarters"),
    ),

    # ---- data_privacy_statute ----
    Fact(
        "priv-01", "data_privacy_statute", "Section 101 - Definitions and Scope",
        "This statute applies to any entity that collects, processes, or stores personal data of more than 5,000 individuals within a 12-month period in connection with commercial activity.",
        "What threshold number of individuals' personal data triggers applicability of this statute?",
        "The statute applies once an entity collects, processes, or stores personal data of more than 5,000 individuals within 12 months.",
        ("5,000 individuals", "12-month"),
    ),
    Fact(
        "priv-02", "data_privacy_statute", "Section 108 - Consent Requirements",
        "Affirmative, opt-in consent must be obtained before collecting sensitive personal data, including health information, biometric identifiers, or precise geolocation data.",
        "What type of consent is required before collecting sensitive personal data such as biometric identifiers?",
        "Affirmative, opt-in consent must be obtained before collecting such sensitive personal data.",
        ("opt-in consent", "biometric", "sensitive personal data"),
    ),
    Fact(
        "priv-03", "data_privacy_statute", "Section 108 - Consent Requirements",
        "Consent requests must be presented separately from other contractual terms and must be revocable by the individual at any time through a mechanism no more burdensome than the one used to obtain consent.",
        "Can an individual revoke consent to data processing, and under what condition regarding the revocation mechanism?",
        "Yes, consent must be revocable at any time through a mechanism no more burdensome than the one used to obtain it.",
        ("revocable", "consent", "no more burdensome"),
    ),
    Fact(
        "priv-04", "data_privacy_statute", "Section 114 - Breach Notification",
        "Upon discovery of a data breach affecting personal data, the covered entity must notify the regulator within 72 hours of becoming aware of the breach.",
        "Within how many hours of discovering a data breach must the regulator be notified?",
        "The regulator must be notified within 72 hours of the entity becoming aware of the breach.",
        ("72 hours", "breach", "regulator"),
    ),
    Fact(
        "priv-05", "data_privacy_statute", "Section 114 - Breach Notification",
        "Affected individuals must be notified of a data breach without undue delay and in no case later than 30 days after the entity becomes aware of the breach, unless a law enforcement investigation requires a delay.",
        "By when must affected individuals be notified of a data breach, absent a law enforcement delay?",
        "Affected individuals must be notified no later than 30 days after the entity becomes aware of the breach.",
        ("30 days", "affected individuals", "notified"),
    ),
    Fact(
        "priv-06", "data_privacy_statute", "Section 114 - Breach Notification",
        "A breach notification to affected individuals must describe the categories of personal data involved, the estimated number of individuals affected, and the remedial steps taken by the entity.",
        "What three elements must a breach notification to affected individuals include?",
        "The categories of personal data involved, the estimated number of individuals affected, and the remedial steps taken.",
        ("categories of personal data", "remedial steps"),
    ),
    Fact(
        "priv-07", "data_privacy_statute", "Section 121 - Data Retention",
        "Personal data may not be retained longer than is necessary for the purpose for which it was collected, and in no case longer than 24 months after the purpose has been fulfilled, absent an independent legal retention obligation.",
        "What is the maximum retention period for personal data after the collection purpose has been fulfilled?",
        "No longer than 24 months after the purpose has been fulfilled, absent an independent legal retention obligation.",
        ("24 months", "retention", "purpose"),
    ),
    Fact(
        "priv-08", "data_privacy_statute", "Section 125 - Individual Rights",
        "An individual has the right to request access to, correction of, or deletion of their personal data, and the covered entity must respond to such a request within 45 calendar days.",
        "Within how many calendar days must an entity respond to an individual's access, correction, or deletion request?",
        "The entity must respond within 45 calendar days.",
        ("45 calendar days", "access", "deletion"),
    ),
    Fact(
        "priv-09", "data_privacy_statute", "Section 125 - Individual Rights",
        "An entity may charge a reasonable fee for a data access request only if the request is manifestly unfounded, excessive, or repetitive.",
        "Under what circumstances may an entity charge a fee for a data access request?",
        "Only if the request is manifestly unfounded, excessive, or repetitive.",
        ("fee", "manifestly unfounded", "excessive", "repetitive"),
    ),
    Fact(
        "priv-10", "data_privacy_statute", "Section 130 - Cross-Border Transfers",
        "Personal data may be transferred outside the jurisdiction only if the receiving country has been determined to provide an adequate level of protection, or if the transfer is subject to standard contractual clauses approved by the regulator.",
        "Under what two conditions may personal data be transferred outside the jurisdiction?",
        "If the receiving country provides an adequate level of protection, or the transfer uses standard contractual clauses approved by the regulator.",
        ("cross-border", "adequate level of protection", "standard contractual clauses"),
    ),
    Fact(
        "priv-11", "data_privacy_statute", "Section 135 - Data Protection Officer",
        "An entity that processes personal data of more than 50,000 individuals annually or that processes sensitive personal data as a core activity must appoint a data protection officer.",
        "At what annual processing volume must an entity appoint a data protection officer?",
        "An entity processing personal data of more than 50,000 individuals annually must appoint a data protection officer.",
        ("50,000 individuals", "data protection officer"),
    ),
    Fact(
        "priv-12", "data_privacy_statute", "Section 140 - Penalties",
        "A violation of this statute may result in an administrative fine of up to 4 percent of the entity's global annual revenue or 20 million currency units, whichever is greater.",
        "What is the maximum administrative fine for a violation of this statute?",
        "Up to 4 percent of the entity's global annual revenue or 20 million currency units, whichever is greater.",
        ("4 percent", "20 million", "fine"),
    ),

    # ---- insurance_policy_group402 ----
    Fact(
        "ins-01", "insurance_policy_group402", "Coverage Overview",
        "This policy covers medically necessary inpatient hospital services, outpatient surgery, emergency care, and physician office visits, subject to the deductibles and copayments described in this document.",
        "What categories of services does this policy cover?",
        "Medically necessary inpatient hospital services, outpatient surgery, emergency care, and physician office visits.",
        ("inpatient", "outpatient surgery", "emergency care"),
    ),
    Fact(
        "ins-02", "insurance_policy_group402", "Deductibles and Copays",
        "The annual individual deductible under this plan is 1,500 dollars, and the annual family deductible is 3,000 dollars, after which coinsurance applies.",
        "What is the annual individual deductible under Group Plan 402?",
        "The annual individual deductible is 1,500 dollars.",
        ("1,500 dollars", "individual deductible"),
    ),
    Fact(
        "ins-03", "insurance_policy_group402", "Deductibles and Copays",
        "After the deductible is met, the plan pays 80 percent of covered in-network charges and the member is responsible for a 20 percent coinsurance up to the annual out-of-pocket maximum.",
        "What percentage of covered in-network charges does the plan pay after the deductible is met?",
        "The plan pays 80 percent of covered in-network charges after the deductible is met.",
        ("80 percent", "coinsurance", "deductible"),
    ),
    Fact(
        "ins-04", "insurance_policy_group402", "Deductibles and Copays",
        "The annual out-of-pocket maximum is 6,000 dollars for an individual and 12,000 dollars for a family, inclusive of deductible, copayments, and coinsurance.",
        "What is the annual out-of-pocket maximum for an individual under this plan?",
        "The annual out-of-pocket maximum for an individual is 6,000 dollars.",
        ("6,000 dollars", "out-of-pocket maximum"),
    ),
    Fact(
        "ins-05", "insurance_policy_group402", "Deductibles and Copays",
        "A copayment of 30 dollars applies to each primary care physician office visit, and a copayment of 60 dollars applies to each specialist office visit.",
        "What is the copayment for a specialist office visit under this plan?",
        "The copayment for a specialist office visit is 60 dollars.",
        ("60 dollars", "specialist"),
    ),
    Fact(
        "ins-06", "insurance_policy_group402", "Exclusions",
        "This policy does not cover cosmetic surgery that is not medically necessary, experimental treatments not approved by the relevant regulatory authority, or services received outside the plan's provider network without prior authorization.",
        "Does this policy cover cosmetic surgery that is not medically necessary?",
        "No, cosmetic surgery that is not medically necessary is excluded from coverage.",
        ("cosmetic surgery", "excluded", "not medically necessary"),
    ),
    Fact(
        "ins-07", "insurance_policy_group402", "Exclusions",
        "Coverage does not extend to services rendered by a provider who is not licensed in the jurisdiction where the service is performed, nor to injuries sustained during the commission of a felony.",
        "What type of provider's services are excluded from coverage under this policy?",
        "Services rendered by a provider who is not licensed in the jurisdiction where the service is performed are excluded.",
        ("not licensed", "excluded"),
    ),
    Fact(
        "ins-08", "insurance_policy_group402", "Prior Authorization",
        "Prior authorization is required for all non-emergency inpatient admissions, advanced imaging such as MRI or CT scans, and any procedure with a billed cost exceeding 5,000 dollars.",
        "What billed cost threshold triggers a prior authorization requirement for a procedure?",
        "A procedure with a billed cost exceeding 5,000 dollars requires prior authorization.",
        ("5,000 dollars", "prior authorization"),
    ),
    Fact(
        "ins-09", "insurance_policy_group402", "Prior Authorization",
        "A prior authorization request must be submitted at least 5 business days before a scheduled non-emergency procedure, and the plan must respond within 3 business days of receiving a complete request.",
        "How many business days before a scheduled non-emergency procedure must a prior authorization request be submitted?",
        "A prior authorization request must be submitted at least 5 business days before the procedure.",
        ("5 business days", "prior authorization request"),
    ),
    Fact(
        "ins-10", "insurance_policy_group402", "Claims Process",
        "A claim for reimbursement must be submitted within 180 days of the date of service, using the standard claim form along with an itemized bill from the provider.",
        "Within how many days of the date of service must a reimbursement claim be submitted?",
        "A reimbursement claim must be submitted within 180 days of the date of service.",
        ("180 days", "claim", "date of service"),
    ),
    Fact(
        "ins-11", "insurance_policy_group402", "Claims Process",
        "The plan must process a clean claim and issue payment or denial within 30 calendar days of receipt; failure to do so entitles the provider to interest on the unpaid amount.",
        "Within how many calendar days must the plan process a clean claim and issue payment or denial?",
        "The plan must process a clean claim within 30 calendar days of receipt.",
        ("30 calendar days", "clean claim"),
    ),
    Fact(
        "ins-12", "insurance_policy_group402", "Appeals",
        "A member who disagrees with a claim denial may file a first-level internal appeal within 180 days of the denial notice, and the plan must issue a decision within 30 calendar days for standard appeals or 72 hours for urgent appeals.",
        "How many days does a member have to file a first-level internal appeal after a claim denial?",
        "A member has 180 days from the denial notice to file a first-level internal appeal.",
        ("180 days", "internal appeal", "denial"),
    ),

    # ---- lab_sop_document_control ----
    Fact(
        "sop-01", "lab_sop_document_control", "Purpose and Scope",
        "This SOP establishes the process for creating, reviewing, approving, distributing, and revising controlled quality documents within the laboratory quality management system.",
        "What does SOP-QA-011 establish a process for?",
        "It establishes the process for creating, reviewing, approving, distributing, and revising controlled quality documents.",
        ("controlled quality documents", "process"),
    ),
    Fact(
        "sop-02", "lab_sop_document_control", "Document Approval",
        "Every controlled document requires review and signature by the document originator, a subject matter expert, and the quality assurance manager before it becomes effective.",
        "Which three roles must review and sign a controlled document before it becomes effective?",
        "The document originator, a subject matter expert, and the quality assurance manager.",
        ("originator", "subject matter expert", "quality assurance manager"),
    ),
    Fact(
        "sop-03", "lab_sop_document_control", "Document Approval",
        "A controlled document may not be released for use until it has been assigned a unique document number and entered into the master document log.",
        "What two things must happen before a controlled document may be released for use?",
        "It must be assigned a unique document number and entered into the master document log.",
        ("unique document number", "master document log"),
    ),
    Fact(
        "sop-04", "lab_sop_document_control", "Periodic Review",
        "Every controlled document must undergo periodic review at least once every 24 months, even if no content changes are proposed, to confirm continued accuracy and relevance.",
        "How often must a controlled document undergo periodic review at minimum?",
        "At least once every 24 months, even if no content changes are proposed.",
        ("periodic review", "24 months"),
    ),
    Fact(
        "sop-05", "lab_sop_document_control", "Change Control",
        "Any proposed change to a controlled document must be submitted through a change control request and evaluated for impact on validated processes before implementation.",
        "What must be submitted before a controlled document can be changed?",
        "A change control request must be submitted and evaluated for impact on validated processes.",
        ("change control request", "validated processes"),
    ),
    Fact(
        "sop-06", "lab_sop_document_control", "Change Control",
        "A change classified as major, meaning it affects a validated method or a critical process parameter, requires re-validation and approval from the quality assurance manager and the laboratory director prior to implementation.",
        "What approvals are required for a major change that affects a validated method?",
        "Approval from the quality assurance manager and the laboratory director, along with re-validation, prior to implementation.",
        ("major change", "re-validation", "laboratory director"),
    ),
    Fact(
        "sop-07", "lab_sop_document_control", "Training",
        "All personnel affected by a revised controlled document must complete documented training on the revision within 10 business days of the document's effective date.",
        "Within how many business days of a document's effective date must affected personnel complete training on the revision?",
        "Affected personnel must complete documented training within 10 business days of the effective date.",
        ("10 business days", "training", "effective date"),
    ),
    Fact(
        "sop-08", "lab_sop_document_control", "Obsolete Documents",
        "Superseded versions of a controlled document must be removed from active use areas immediately upon the new version's effective date and archived for a minimum of 10 years.",
        "For how many years must superseded versions of a controlled document be archived?",
        "Superseded versions must be archived for a minimum of 10 years.",
        ("superseded", "archived", "10 years"),
    ),
    Fact(
        "sop-09", "lab_sop_document_control", "Obsolete Documents",
        "An obsolete document retained for reference must be clearly stamped or watermarked as OBSOLETE and stored separately from current controlled documents.",
        "How must an obsolete document retained for reference be marked?",
        "It must be clearly stamped or watermarked as OBSOLETE.",
        ("obsolete", "stamped", "watermarked"),
    ),
    Fact(
        "sop-10", "lab_sop_document_control", "Electronic Records",
        "Electronic controlled documents must be maintained in a validated document management system with audit trail functionality that records the identity of the user, the date, and the nature of each change.",
        "What three items must an audit trail record for each change to an electronic controlled document?",
        "The identity of the user, the date, and the nature of each change.",
        ("audit trail", "identity of the user", "nature of each change"),
    ),
    Fact(
        "sop-11", "lab_sop_document_control", "Deviations",
        "Any deviation from an approved controlled document must be documented on a deviation report within 24 hours of discovery and reviewed by the quality assurance manager.",
        "Within how many hours of discovery must a deviation from a controlled document be documented?",
        "A deviation must be documented within 24 hours of discovery.",
        ("deviation", "24 hours"),
    ),
    Fact(
        "sop-12", "lab_sop_document_control", "Roles and Responsibilities",
        "The quality assurance manager holds final authority to approve, reject, or place on hold any controlled document within the quality management system.",
        "Who holds final authority to approve, reject, or place a controlled document on hold?",
        "The quality assurance manager holds final authority over controlled documents.",
        ("quality assurance manager", "final authority"),
    ),
]


def facts_for_doc(doc_id: str) -> list[Fact]:
    return [f for f in FACTS if f.doc_id == doc_id]


def sections_for_doc(doc_id: str) -> list[str]:
    seen: list[str] = []
    for f in facts_for_doc(doc_id):
        if f.section not in seen:
            seen.append(f.section)
    return seen
