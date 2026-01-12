#import "@preview/clear-iclr:0.7.0": iclr2025
#import "@preview/wordometer:0.1.5": total-words, word-count
#import "@preview/codly:1.3.0"
#import "@preview/codly-languages:0.1.10"

#set par.line(numbering: "1")
// Lighter, cleaner “journal PDF” code style using the brand greys as *structure*,
// but keeping the code ink mostly black for print.

// Brand greys (from your brandbook set)
#let g0 = rgb("#E0E1E7")
#let g1 = rgb("#BDC2C8")
#let g2 = rgb("#A5B0C1")
#let accent = rgb("#6E83A2")

// Make a *very* light background by mixing toward white.
#let bg = rgb("#F7F8FB")        // lighter than g0; reads like paper
#let border = g1

#show raw.where(block: true): it => block(
  fill: bg,
  stroke: (paint: border, thickness: 0.5pt),
  radius: 4pt,
  inset: 7pt,
  width: 100%,
  text(font: "DejaVu Sans Mono", size: 9pt, it),
)
#show raw: set text(font: "DejaVu Sans Mono")

#let authors = (
  (
    names: ([Patrick Mogenson],),
    affilation: [
      PumasAI Inc., USA
    ],
    email: "patrick@pumas.ai",
  ),
  (
    names: ([Andreas Noack],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([Mohamed Tarek],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([David Müller-Widmann],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([Julius Krumbiegel],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([Michael Hatherly],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([Niklas Korsbo],),
    affilation: [
      PumasAI Inc., USA
    ],
  ),
  (
    names: ([Christopher Rackauckas],),
    affilation: [
      PumasAI Inc., USA \
      JuliaHub Inc., USA \
      Massachusetts Institute of Technology, USA
    ],
  ),
  (
    names: ([Vijay Ivaturi],),
    affilation: [
      PumasAI Inc., USA \
      Center for Pharmacometrics, MCOPS, Manipal Academy of Higher Education, India
    ],
    email: "vijay@pumas.ai",
  ),
)


#show: iclr2025.with(
  title: [Pumas: A Unified Platform for Pharmaceutical Modeling and Simulation],
  authors: authors,
  keywords: (
    "Pumas",
    "pharmacometrics",
    "NLME",
    "PKPD",
    "Julia",
    "unified platform",
    "model-informed drug development",
  ),
  abstract: [
    // #word-count(total => [
    Pumas (Pharmaceutical Modeling and Simulation) is a software ecosystem for quantitative pharmaceutical analytics built on the Julia programming language.
    This article introduces the Pumas ecosystem and demonstrates its integrated workflow through a complete case study of a multiple ascending dose trial with pharmacokinetic and pharmacodynamic endpoints.

    The platform integrates nine capability domains: CDISC-compliant data preparation, exploratory data analysis with publication-quality visualization,
    non-compartmental analysis with bioequivalence testing, nonlinear mixed effects modeling with multiple estimation algorithms (FOCE, Laplacian, SAEM, MCEM, Bayesian),
    simulation with differential equations using a variety of solvers, machine learning integration via DeepPumas, optimal experimental design,
    parallel computing infrastructure, and automated reporting through Quarto integration.

    We demonstrate the unified workflow from data preparation through model-based dose selection. Using a case study with 60 subjects across 6 dose levels,
    we develop a two-compartment pharmacokinetic model with covariate effects (sex on clearance, weight on volume), extend to pharmacodynamics with indirect response modeling (Kout inhibition with effect compartment),
    and simulate alternative dosing regimens to evaluate target attainment metrics.
    The complete analysis executes within a single computational framework, eliminating tool-switching overhead and data conversion steps.

    Pumas's design principles—multiple dispatch, composability, and performance—enable both rigorous statistical modeling and rapid iterative exploration.
    Integration across workflow stages reduces friction in pharmacometric analyses, supporting quantitative decision-making from first-in-human studies through regulatory submission.
    The platform supports Phase 1-3 clinical trial analysis and model-informed drug development applications.

    // [Words: #total.words

    //     Characters: #total.characters] <no-wc>
    // ], exclude: <no-wc>
    )
  ],
  bibliography: bibliography("references.bib"),
  appendix: [
    #set page(background: rotate(24deg, text(100pt, fill: rgb("#33201d1f"))[
      *DRAFT*
    ]))

    = Complete Annotated Workflow and Implementation Details

    This appendix contains the complete code and detailed explanations for all analyses presented in the main text.
    Each section corresponds to a stage in the pharmacometric workflow and provides reproducible examples with full parameter specifications.

    == A.1 Data Simulation Code

    Complete code for generating the synthetic multiple ascending dose dataset used throughout the article.
    Includes model definition, population generation, and data export. See lines 1-307 in `sections/10_report.qmd`.

    == A.2 Data Preparation and Exploratory Analysis

    Detailed data manipulation workflows using DataFramesMeta.jl, population object creation with `read_pumas()`,
    and comprehensive exploratory data analysis including summary tables and concentration-time profile visualization.
    See lines 308-1061 in `sections/10_report.qmd`.

    == A.3 Non-Compartmental Analysis

    Complete NCA workflow including `NCAPopulation` creation, exposure metric calculation, dose proportionality assessment,
    and statistical testing with power models. See lines 1062-1257 in `sections/10_report.qmd`.

    == A.4 Complete Model Definitions

    Full code for all model specifications: one-compartment, two-compartment, covariate models, and multi-endpoint PK-PD models.
    Includes detailed block-by-block explanations and parameter initialization strategies. See lines 1258-1903 in `sections/10_report.qmd`.

    == A.5 Initial Validation and Prior Predictive Checks

    Procedures for validating model specification using `loglikelihood()`, `findinfluential()`, and prior predictive simulation.
    See lines 1320-1356 in `sections/10_report.qmd`.

    == A.6 Model Fitting Workflows

    Progressive estimation strategies (NaivePooled → FOCE), convergence diagnostics, and handling of estimation challenges.
    See lines 1357-1593 in `sections/10_report.qmd`.

    == A.7 Extended Diagnostics

    Comprehensive diagnostic procedures including individual subject fits with pagination, empirical Bayes estimate shrinkage analysis,
    and influence diagnostics. See lines 1384-1763 in `sections/10_report.qmd`.

    == A.8 Model Comparison Procedures

    Formal model comparison using `compare_estimates()`, `metrics_table()`, information criteria (AIC, BIC),
    and likelihood ratio testing for nested models. See lines 1621-1763 in `sections/10_report.qmd`.

    == A.9 Covariate Modeling Details

    Complete covariate identification workflow using EBE diagnostics, implementation patterns for categorical and continuous covariates,
    and evaluation of covariate model adequacy. See lines 1765-2003 in `sections/10_report.qmd`.

    == A.10 Parameter Uncertainty Quantification

    Methods for uncertainty quantification using `infer()` with sandwich estimator, bootstrap procedures,
    sampling-importance-resampling (SIR), and interpretation of variance-covariance matrices.
    See lines 1987-2003 in `sections/10_report.qmd`.

    == A.11 PK-PD Modeling (Simultaneous and Sequential)

    Detailed multi-endpoint modeling approaches including simultaneous PK-PD estimation, sequential modeling with parameter transfer,
    indirect response mechanisms, and exposure-response analysis. See lines 2004-2408 in `sections/10_report.qmd`.

    == A.12 Simulation and Decision Support

    Extended simulation scenarios, virtual population generation, target attainment metric calculation,
    sensitivity analyses, and interpretation guidelines for dose selection. See lines 2409-2640 in `sections/10_report.qmd`.

    == A.13 Installation and Setup

    Instructions for installing Julia, Pumas platform components, VS Code integration, and Quarto setup.
    Package version management with Project.toml and Manifest.toml for reproducible environments.

    == A.14 Additional Resources

    Links to documentation (docs.pumas.ai), tutorials (tutorials.pumas.ai), community forums, GitHub repositories,
    and training materials. Includes guidance on accessing AskPumas AI assistant for interactive learning.
  ],
  accepted: none,
)

#let url(uri) = link(uri, raw(uri))

// #vruler(offset: -1.7in)
#import "@preview/equate:0.2.1": equate
#show: equate.with(breakable: true, sub-numbering: true, number-mode: "line")
#set math.equation(numbering: "(1.1)", supplement: "Eq.")

#show: word-count

= Word count

#total-words words, excluding the appendix and abstract

#set page(background: rotate(24deg, text(100pt, fill: rgb("#33201d1f"))[
  *DRAFT*
]))
= Introduction
<sec-introduction>

== The Pharmaceutical Analytics Challenge

Modern drug development requires quantitative decision-making across discovery, clinical development, and regulatory submission.
A single pharmacometric analysis integrates data manipulation, exploratory visualization, statistical modeling, simulation, and regulatory reporting—tasks
that historically required five or more software packages. Data preparation occurs in R or SAS, nonlinear mixed effects modeling in NONMEM @beal2013nonmem or Monolix @monolix2024,
simulation in Python or MATLAB, non-compartmental analysis in specialized tools like WinNonlin, and final reporting in separate systems.

This fragmentation introduces systematic inefficiencies. Data format conversions between tools create opportunities for errors.
Context switching between programming languages and interfaces slows iterative model refinement. Reproducing analyses across tools requires
maintaining compatibility across multiple software versions and file formats. Each tool boundary represents a potential failure point in the analytical workflow.

== The Pumas Platform: Design Philosophy

Pumas (Pharmaceutical Modeling and Simulation) integrates the complete pharmaceutical analytics lifecycle in a single computational framework built on
the Julia programming language @Julia-2017. The platform design rests on four core principles:

*Single-Language Integration*: All components are native to Julia, eliminating language boundaries between workflow stages.
Data manipulation, statistical modeling, differential equation solving, and visualization execute within the same runtime environment.

*Types and Multiple Dispatch*: Using Julia's type system to organize input and output into data structures to error at object construction if assumptions are violated and to use a coherent workflow by allowing functions to dispatch to the correct methods based on their types. Unlike legacy tools that may silently ignore errors or data inconsistencies until runtime failure, Pumas enforces constraints immediately.
Population objects validate monotonic time, observation-model matching, and covariate completeness when created.
Model definitions enforce parameter domain constraints before estimation begins. Multiple dispatch allows the user to use similar script templates, few functions and rely on multiple dispatch to use type information to calculate and return the relevant result for a variety of input.

*Composability*: Functions chain naturally—outputs from one stage become inputs to the next without manual conversion.
The same ```julia Population``` object serves model fitting, and simulation.
The same ```julia @model``` specification handles initial validation, parameter estimation, diagnostics, and scenario evaluation.

*Performance*: Julia's just-in-time compilation resolves the "two-language problem" @Julia-2017, allowing models to be defined and executed in the same high-level language without transpilation to C++ or Fortran.
This architecture enables aggressive optimization while maintaining code readability. Automatic differentiation enables efficient gradient-based optimization. Parallel computing infrastructure scales across CPU cores without explicit user configuration.

These principles enable reproducible workflows through version-controlled scripts, Quarto notebook integration, and automated reporting.
The entire analysis from raw data to regulatory submission documents can be rendered into a single reproducible report.

== Ecosystem Capabilities: Complete Pharmaceutical Analytics

=== Software Ecosystem and Package Architecture

The Pumas platform combines proprietary pharmaceutical modeling capabilities with open-source Julia packages.
This architecture enables specialized pharmacometric functionality while leveraging the broader scientific computing ecosystem.

*Pumas Platform Components* (proprietary, requiring license):
- *Pumas*: Core pharmacometric modeling engine including ```julia @model``` macro, NLME estimation algorithms (FOCE, Laplace, SAEM, MCEM, Bayesian), population objects, and diagnostics
- *PumasUtilities*: Pharmacometric plotting functions (```julia goodness_of_fit()```, ```julia vpc_plot()```, ```julia subject_fits()```, ```julia empirical_bayes_vs_covariates()```)
- *DeepPumas*: Neural network integration for scientific machine learning and DeepNLME
- *PharmaDatasets.jl*: Curated pharmaceutical datasets for learning and method validation

*Open-Source Julia Packages* (developed and maintained by PumasAI, available on GitHub):
- *SummaryTables.jl* @SummaryTables: Publication-quality tables for demographics, data summaries, and regulatory submissions
- *WriteDocx.jl* @WriteDocx: Programmatic generation of Microsoft Word documents for reporting
- *AlgebraOfGraphics.jl* @AlgebraOfGraphics: Declarative grammar of graphics for visualization (similar to ggplot2 @wickham2016ggplot2 in R)
- *QuartoNotebookRunner.jl* @QuartoNotebookRunner: Backend for executing Quarto notebooks with Julia
- *QuartoTools.jl* @QuartoTools: Integration utilities for Quarto document rendering
- *SimpleChains.jl* @SimpleChains: High-performance neural networks for small models

*Open-Source Julia Ecosystem* (a few examples; Pumas generally interoperates with the Julia ecosystem subject to some version constraints imposed by a given Pumas release for stability):
- *DataFrames.jl* @DataFrames: Tabular data structures (similar to pandas in Python or data.frame in R)
- *DataFramesMeta.jl* @DataFramesMeta: Macro-based data manipulation (similar to dplyr @wickham2023dplyr in R)
- *CSV.jl* @CSV: CSV file reading and writing
- *CairoMakie.jl* / *Makie.jl* @danisch2021makie: Publication-quality graphics rendering
- *DifferentialEquations.jl* @rackauckas2017differentialequations: Differential equation solvers (ODE, SDE, DAE)
- *AdvancedHMC.jl* @AdvancedHMC: Hamiltonian Monte Carlo for Bayesian inference
- *StableRNGs.jl* @StableRNGs: Reproducible random number generation

*External Tools*:
- *Quarto* @allaire2023quarto: Scientific publishing system for reproducible documents (not a Julia package; separate installation)

This layered architecture means users benefit from continuous improvements in the Julia ecosystem while accessing specialized pharmacometric capabilities through the Pumas platform.
Open-source components can be used independently for general data science tasks; the proprietary components provide domain-specific modeling and regulatory-ready outputs.

The Pumas ecosystem encompasses nine integrated capability domains that span the drug development lifecycle:

=== Data Preparation and Standards

*CDISC-Compliant Workflows*: Data preparation through ADaM.jl creates analysis datasets following Clinical Data Interchange Standards Consortium (CDISC) specifications @cdisc2023
for both non-compartmental analysis and population pharmacokinetic/pharmacodynamic modeling. CDISC compliance ensures regulatory acceptability and cross-study consistency.

*Data Format Support and Data Representation*:  All the usual file formats such as .csv, .xlsx and  .sas7bdat are supported through packages `CSV.jl` and `ReadStatTables.jl` and are converted into `DataFrame`s.  The ```julia read_pumas() ``` function converts the `DataFrame`s into validated ```julia Population``` objects.
Validation checks include monotonic time within subjects, observation-model matching, and covariate completeness.
Errors are caught at data construction with informative messages indicating specific subjects and time points requiring correction.

*Integration with Julia Data Ecosystem*: Built on DataFrames.jl (tabular data structures) and DataFramesMeta.jl (macro-based transformations similar to dplyr @wickham2023dplyr in R).
Most Pumas objects convert to DataFrames for custom analyses or export to external tools.

*Example Datasets*: PharmaDatasets.jl provides standard pharmaceutical datasets for learning and method validation,
including published PK/PD studies with documentation of data structure and details of data source.

=== Exploratory Data Analysis

*Statistical Foundations*: Built on Statistics.jl (descriptive statistics, hypothesis tests) and GLM.jl (generalized linear models for exploratory regression).
Standard statistical tests (t-tests, ANOVA, correlation) available for covariate analysis and data quality assessment.

*Visualization*: Integrated plotting through AlgebraOfGraphics.jl (an algebra of graphics approach) and CairoMakie.jl  (publication-quality rendering).
Algebra-of-graphics enables declarative plot specification: map data to aesthetics, apply geometric representations, facet by variables, similar to ggplot2 in R.
Plots are Julia objects that can be composed, modified, and exported to vector formats.

*Summary Tables*: SummaryTables.jl generates publication-quality demographic tables, data quality summaries, and listing tables matching regulatory submission standards.
Tables support stratification by categorical variables, custom summary functions, and formatting specifications for decimal places and missing data handling.

*Pre-Modeling Exploration*: Automated generation of concentration-time profiles stratified by dose and time, covariate distribution visualization,
and identification of missing data patterns or outliers requiring investigation.

=== Noncompartmental Analysis

*Core NCA*: The ```julia run_nca()``` function calculates standard exposure metrics including area under the curve (AUC), maximum concentration (Cmax),
time of maximum concentration (Tmax), terminal elimination half-life (t½), apparent clearance (CL/F), and apparent volume of distribution (Vz/F).
Calculation methods follow FDA @fda2014bioequivalence and EMA @ema2010bioequivalence guidances.

*Data Handling*: Integrated support for missing observations and below-limit-of-quantification (BLQ) data.
BLQ handling options include: treat as zero, treat as missing, or impute using LLOQ/2. Missing data handling follows established pharmacokinetic principles
(e.g., no extrapolation beyond last quantifiable concentration for AUClast).

*Superposition*: Predict multiple-dose steady-state profiles from single-dose pharmacokinetics using linear superposition principles

*Automation*: NCA report generation with configurable parameter sets. Users specify which parameters to calculate (e.g., exclude Vz/F if terminal phase poorly characterized).
Reports include parameter estimates, confidence intervals, and dose normalization for proportionality assessment.

=== Dose Proportionality

Statistical evaluation of dose linearity using power models. The ```julia NCA.DoseLinearityPowerModel()``` function fits the regression model $log(Y) = alpha + beta dot log("Dose")$ where Y is an NCA parameter (AUC, Cmax) and tests whether the slope β differs significantly from 1.0 (perfect proportionality). The function accepts an ```julia NCAReport``` and parameter name (e.g., ```:auc_0_inf```, ```:cmax```), returning a ```julia DoseLinearityPowerModel``` object containing point estimate and confidence interval for β.

=== Bioequivalence Analysis

Statistical assessment of bioequivalence through comprehensive testing procedures that meet regulatory submission requirements. The platform supports parametric methods including the two one-sided tests (TOST) procedure as well as non-parametric alternatives for non-normal distributions. Study designs handled include standard 2×2 crossover designs alongside more complex arrangements: higher-order crossover, parallel group, and replicate designs. For highly variable drugs, reference-scaled average bioequivalence (RSABE) calculations adjust equivalence bounds based on within-subject variability of the reference product. The workflow generates geometric mean ratios with 90% confidence intervals, automatically applying log-transformation to pharmacokinetic metrics like AUC and Cmax while accommodating untransformed analyses when appropriate. Equivalence bounds typically follow regulatory guidance of 80-125%, though the platform allows customization for specific cases. Integration with the NCA workflow creates a seamless path from exposure metric calculation via ```julia run_nca()``` to bioequivalence testing through ```julia pumas_be()```, with automatic handling of period effects, sequence effects, and subject-level random effects. Built-in templates encode common study designs and regulatory expectations from both FDA and EMA guidelines, reducing setup time and ensuring compliance with jurisdictional requirements.

=== Sample Size and Power

Prospective power calculation for bioequivalence studies using ```julia samplesize()``` enables sample size determination before trial execution. Given expected within-subject variability (coefficient of variation), anticipated geometric mean ratio, equivalence bounds (typically 80-125%), and desired power level (conventionally 80% or 90%), the function calculates required sample size per sequence. The calculation accounts for study design (2×2 crossover, parallel, replicate), significance level (α, typically 0.05 for 90% confidence intervals), and dropout rates. Higher within-subject variability demands larger sample sizes to achieve adequate power—a CV of 30% requires substantially more subjects than 15% to detect bioequivalence with the same power. For highly variable drugs (CV > 30%), reference-scaled average bioequivalence may reduce sample size requirements by widening acceptance criteria based on reference product variability. Sensitivity analyses across plausible CV ranges inform realistic sample size planning and budget allocation for Phase 1 bioequivalence trials.

=== Nonlinear Mixed Effects (NLME) Modeling

*Model Definition*: Declarative specification using the ```julia @model``` macro with dedicated functional blocks:
- ```julia @param```: Population parameters with domain constraints (bounds, positivity, initial values) and distributional priors
- ```julia @random```: Multivariate distributions for inter-subject variability and inter-occasion variability
- ```julia @covariates```: Explicit declaration of subject-level covariates used in parameter derivations
- ```julia @pre```: Individual parameter calculations incorporating covariates and random effects
- ```julia @dynamics```: Differential equations or closed-form solutions for system dynamics
- ```julia @derived```: Observation models with error structures
- ```julia @vars```: Intermediate variables for code clarity (optional)

This block structure enforces logical organization and makes model assumptions explicit. There are several other blocks not covered here, please refer to the Pumas documentation @pumas2024docs for a complete list

*Data Type Support*: Continuous data (plasma concentrations, biomarkers), count data (seizure frequency, tumor lesion counts),
categorical data (response classifications), time-to-event data (survival analysis), and combinations enabling multi-endpoint modeling
(e.g., PK concentrations with efficacy or safety endpoints).

*Estimation Algorithms*:
- *Classical*: Naive Pooled (fixed effects only), First Order (FO), First Order Conditional Estimation @lindstrom1990nonlinear (FOCE with or without interaction), Laplacian approximation
- *Maximum A Posteriori*: MAP (individual parameter estimation) and JointMAP (simultaneous population and individual estimation)
- *Expectation-Maximization*: SAEM @kuhn2005maximum (Stochastic Approximation EM, handles complex random effect structures), MCEM (Monte Carlo EM)
- *Bayesian*: Full posterior inference via Hamiltonian Monte Carlo @hoffman2014nuts through AdvancedHMC.jl integration, enabling credible intervals and posterior predictive checks

Algorithm selection depends on model complexity, data richness, and inference goals. SAEM handles models intractable with FOCE and Bayesian methods enable prior incorporation and posterior predictive validation.

*Model Diagnostics*: Comprehensive suite including goodness-of-fit plots (observations vs predictions, residuals vs predictions/time),
visual predictive checks @holford2005vpc @bergstrand2011vpc (VPC with stratification), individual subject fits with empirical Bayes estimates,
influence diagnostics (identifying subjects with large impact on parameter estimates), and EBE shrinkage analysis @savic2009shrinkage (assessing information content for random effects) and many others.

*Parameter Inference*: Uncertainty quantification via ```julia infer()``` function with three methods:
- *Sandwich estimator* @white1980sandwich: Fast asymptotic standard errors from observed Fisher information (default) or the inverse Hessian.
- *Bootstrap* @efron1994bootstrap: Non-parametric uncertainty via resampling subjects with replacement, with or without stratification
- *Sampling-Importance-Resampling (SIR)*: Samples from asymptotic distribution weighted by likelihood ratios, providing non-asymptotic confidence intervals
- *MarginalMCMC*: Infer the uncertainty of the parameters of the fitted model fpm by sampling from the marginal likelihood with the MCMC algorithm

*Automation*: Automated covariate selection using stepwise forward/backward procedures using the ```julia covariate_select()``` function. Starting from a base model, the algorithm tests covariate effects sequentially,
retaining covariates that improve model fit beyond a specified threshold (e.g., ΔOFV > 3.84 for forward selection, > 10.83 for backward elimination).
Automation reduces manual model-building time but requires validation of selected covariates against physiological plausibility.

*Optimization*: Automatic detection of closed-form solutions for standard linear compartmental models (e.g., one-compartment, two-compartment, three-compartment with various absorption patterns).
When closed-form solutions exist, Pumas uses analytical expressions via the matrix exponentials rather than numerical ODE solving, reducing computation time by orders of magnitude.

=== Simulation and Advanced Modeling

*Differential Equation Integration*: Deep integration with DifferentialEquations.jl @rackauckas2017differentialequations provides access to over 100 ODE, DAE (differential-algebraic),
and SDE (stochastic differential equation) solvers. Features include adaptive timestepping (automatically adjusts step size for accuracy and stability),
automatic stiffness detection (switches between non-stiff and stiff solvers as needed), and event handling (implements dosing, reset conditions, or discontinuities).

*Clinical Trial Simulation*: Generate virtual populations with specified covariate distributions, simulate dosing regimens using ```julia DosageRegimen()```
(supports complex schedules: loading doses, multiple compartments, time-varying infusions), and evaluate trial outcomes under protocol variations.
Applications include dose selection, sample size determination, and protocol optimization.

*Sensitivity Analysis*: Global sensitivity analysis identifies parameters with largest impact on model outputs across parameter space using methods like
Sobol indices @sobol2001sensitivity, Morris screening @morris1991factorial, and eFast methods @saltelli1999efast. Local sensitivity analysis computes gradients at specific parameter values to understand output changes with small parameter perturbations.

*Optimal Experimental Design* @atkinson2007optimal @mentre1997optimal @nyberg2012methods: Tools for designing efficient PK/PD studies by optimizing sampling times, dose levels, or subject allocation.
Criteria include D-optimal (maximize determinant of Fisher information matrix), ED-optimal (maximize determinant of expected Fisher information),
and A-optimal (minimize trace of covariance matrix). Applications include Phase 1 dose escalation design and Phase 2 dose selection.

*Parallel Simulation*: Automatic parallelization across subjects and simulation replicates using Julia's multithreading and distributed computing.

=== Scientific Machine Learning and DeepNLME

*DeepPumas*: Integrates neural networks with mechanistic pharmacokinetic/pharmacodynamic models in a NLME framework with random effects (DeepNLME) included. Applications include:
- Replace unknown mechanistic components with neural networks (universal differential equations @rackauckas2020universal)
- Transfer learning: train models on preclinical data, fine-tune on clinical data

DeepPumas combines automatic differentiation through neural networks and differential equations @raissi2019pinn, enabling end-to-end training with gradient-based optimization.

=== Agentic AI Workflows

*PumasAide*: Agentic AI assistant for automated modeling workflows. Capabilities include suggesting model structures based on data characteristics,
automatically generating diagnostic plots with interpretations, and proposing covariate model refinements based on EBE analysis.
PumasAide accelerates modeling for experienced users and provides guidance for learners.

=== Visualization and Reporting

*Pharmacometric Plots*: Built-in functions for standard diagnostic visualizations:
- Goodness-of-fit: ```julia goodness_of_fit()``` generates four-panel plot (observations vs population predictions, observations vs individual predictions, conditional weighted residuals vs population predictions, conditional weighted residuals vs time)
- Visual predictive checks: ```julia vpc()``` simulates replicates, ```julia vpc_plot()``` overlays observed and predicted quantiles with confidence intervals
- Individual fits: ```julia subject_fits()``` displays observed data with individual predictions, supports pagination for large datasets
- EBE diagnostics: ```julia empirical_bayes_vs_covariates()``` plots random effects against covariates to identify covariate relationships

*Custom Visualization*: Full access to AlgebraOfGraphics.jl and Makie.jl for publication-quality custom graphics.
Algebra-of-graphics approach enables layered plot construction: specify data mappings, add geometric elements, apply statistical transformations, customize aesthetics.
Output formats include PNG, PDF, SVG for publications.

*Automated Reporting*: Quarto @allaire2023quarto integration combines code, results, and narrative in single reproducible documents.
Quarto notebooks execute Julia code blocks, capture outputs (tables, plots), and render to HTML, PDF, or Word formats.
Version-controlled notebooks ensure analysis reproducibility @sandve2013reproducible and facilitate collaborative model development.

*Regulatory-Ready Outputs*: Export formats compatible with regulatory submission requirements including SDTM/ADaM datasets,
tables/listings/figures matching submission standards, and model documentation with parameter estimates, covariance matrices, and diagnostic plots.

=== Performance and Deployment

*Parallel Computing*: First-class support for multithreading (within-node parallelism across CPU cores) and distributed computing (across-node parallelism across cluster or cloud nodes).
Julia's parallel constructs enable transparent parallelization—many Pumas functions automatically parallelize over subjects or simulation replicates when multiple cores are available.

*Deployment Flexibility*:
- Desktop: macOS, Windows with identical functionality across platforms
- HPC Clusters: Integration with job schedulers (SLURM)
- Cloud: Kubernetes deployment for elastic scaling—automatically provision compute resources based on workload

=== User Interfaces and Learning Resources

*PumasCP*: Graphical user interface for common analyses requiring no coding:
- Non-compartmental analysis with automated report generation
- Bioequivalence testing with parametric and non-parametric methods
- Dose proportionality assessment with power models and visualization
- Superposition analysis for multiple-dose prediction

PumasCP lowers entry barrier for users preferring graphical interfaces while maintaining full programmatic access for advanced users.

*AskPumas*: AI-powered learning assistant providing @askpumas2024:
- Interactive documentation search (natural language queries return relevant documentation sections)
- Code example generation (request examples for specific analyses, receive working code)
- Error diagnosis and troubleshooting (paste error messages, receive explanations and solutions)

*Tutorials*: Comprehensive tutorial collection at tutorials.pumas.ai covering:
- Getting started: installation, first PK model, basic diagnostics
- Intermediate topics: covariate modeling, multi-endpoint analysis, VPC interpretation
- Advanced workflows: optimal design, Bayesian inference, machine learning integration
- Domain-specific: oncology, pediatrics, renal impairment

Tutorials include data files, complete code, and expected outputs for self-paced learning.

*Community Resources*: Active user forums for questions and discussions @pumasdiscourse, GitHub repositories with example workflows and contributed packages,
and regular training workshops (virtual and in-person).

== Article Scope and Organization

This article introduces the Pumas platform architecture (Section 2) and demonstrates its integrated workflow through a complete case study:
a multiple ascending dose study with pharmacokinetic and pharmacodynamic endpoints (Sections 3-7).

We demonstrate:
- Data preparation and exploratory analysis (Section 3)
- Model specification using the ```julia @model``` macro (Section 4)
- Population pharmacokinetic modeling with diagnostics and refinement (Section 5)
- Multi-endpoint PK-PD modeling with indirect response dynamics (Section 6)
- Simulation-based decision support for dose selection (Section 7)

Section 8 discusses platform integration benefits, performance characteristics, extensibility, and current capabilities.

The complete annotated workflow appears in the Appendix with 14 sections (A.1-A.14) providing full code, implementation details,
and alternative approaches. Throughout the article, we reference specific Appendix sections for readers seeking additional depth or implementation guidance.

== Target Audience and Prerequisites

*Primary Audience*: Pharmacometricians and quantitative clinical pharmacologists familiar with compartmental modeling and nonlinear mixed effects concepts.
Readers experienced with NONMEM, Monolix, or Phoenix will find familiar concepts expressed in Pumas syntax.

*Secondary Audience*: Computational scientists, biostatisticians, and pharmaceutical scientists entering pharmacometrics.
Background in differential equations, maximum likelihood estimation, or Bayesian inference is helpful but not required.

*Prerequisites*:
- Basic pharmacokinetics @rowland2011clinical (compartmental models, clearance, volume of distribution, bioavailability concepts)
- Elementary statistics (likelihood, random effects, residual error, confidence intervals)
- Familiarity with programming concepts (functions, data structures, control flow)

Julia syntax is explained as encountered. Full Julia language documentation is available at docs.julialang.org @juliadocs
Users familiar with MATLAB, Python, or R will find Julia syntax recognizable.


// flowchart TB
//     subgraph Stage["Stage"]
//         direction TB
//         S3(["Section 3: Data & EDA"])
//         S4(["Section 4: Model Spec"])
//         S5(["Section 5: PopPK Fitting & Diagnostics"])
//         D{Adequate?}
//         S6(["Section 6: Multi-Endpoint PK-PD Modeling"])
//         S7(["Section 7: Simulations for Decision Support"])

//         S3 --> S4 --> S5 --> D
//         D -->|No| S4
//         D -->|Yes| S6 --> S7
//     end

//     subgraph Func["Functions"]
//         direction TB
//         F3["CSV.read<br/>read_pumas<br/>observations_vs_time<br/>..."]
//         F4["@model macro"]
//         F5["fit · inspect<br/>goodness_of_fit · vpc<br/>compare_estimates · infer<br/>..."]
//         F6["Multi-endpoint model<br/>Indirect response dynamics<br/>..."]
//         F7["DosageRegimen<br/>Subect<br/>simobs · sim_plot<br/>..."]
//     end

//     S3 -.- F3
//     S4 -.- F4
//     S5 -.- F5
//     S6 -.- F6
//     S7 -.- F7
#figure(
  image("figures/fig1-workflow.png", width: 100%),
  caption: [Overview of the Pumas pharmacometric workflow. The upper panel depicts the analysis stages corresponding to Sections 3–7, including the iterative
    model refinement cycle. The lower panel lists key Pumas functions used at each stage.],
) <fig-workflow>


= Case Study Introduction and Data Preparation
<sec-data-prep>

We demonstrate the Pumas integrated workflow through a complete multiple ascending dose (MAD) study analysis.
This section introduces the study design, data loading procedures, population object creation, and exploratory data analysis.

== Multiple Ascending Dose Study Design

*Study Objective*: Develop a pharmacokinetic-pharmacodynamic model to evaluate alternative dosing regimens and support dose selection
for Phase 2 clinical development.

*Study Design*:
- 60 subjects randomized across 6 dose levels: placebo, 100, 200, 400, 800, and 1600 mg
- Dosing schedule: Once daily (QD) for 6 days
- Study period: 10 days total (6 dosing days plus follow-up)

*Observations*:
- *Pharmacokinetics*: Plasma drug concentrations with rich sampling on Days 1 and 6 (0, 0.1, 0.5, 1, 2, 4, 8, 12, 18, 24 hours post-dose),
  sparse sampling on Days 2-5 (trough concentrations)
- *Pharmacodynamics*: Biomarker response measured at selected time points with focus on Days 6-10 to assess steady-state and washout dynamics.
  Therapeutic range: 75-125

*Covariates*: Sex (male/female), baseline body weight (kg), dose level (mg), treatment group

*Data Source*: Simulated dataset (`intro_paper_data.csv`) designed to exhibit realistic PK and PD characteristics.
Simulation code provided in Appendix A.1. Data available as supplementary material.

== Loading and Structuring Data

The analysis begins with loading data from CSV format and inspecting its structure.

```julia
using Pumas, CSV, DataFrames, DataFramesMeta
using PumasUtilities, AlgebraOfGraphics, CairoMakie

# Load data from CSV
path = joinpath(@__DIR__, "data", "intro_paper_data.csv")
wide_data = CSV.read(path, DataFrame)

# Inspect data structure
first(wide_data, 5)
```

The dataset follows a standard pharmacometric format with columns:
- `ID`: Subject identifier
- `NOMTIME`: Nominal time in hours since first dose
- `PROFTIME`: Profile time (hours since most recent dose)
- `PROFDAY`: Profile day (study day, 1-indexed)
- `AMT`: Dose amount in mg (non-missing for dosing events)
- `EVID`: Event type (1 = dosing, 0 = observation)
- `CMT`: Compartment number (1 = depot for oral dosing)
- `ROUTE`: Route of administration ("ev" = extravascular/oral)
- `CONC`: PK concentration in ng/mL (missing for dosing events)
- `PD_CONC`: PD biomarker response in IU/L (missing for dosing events and many PK sampling times)
- `SEX`: Subject sex ("Male" or "Female")
- `WEIGHTB`: Baseline body weight in kg
- `DOSE`: Dose level in mg (0 for placebo)
- `TRTACT`: Treatment group label ("Placebo" or "XXX mg")

This wide-format structure (separate columns for CONC and PD_CONC) enables straightforward multi-endpoint modeling.
Pumas does not supports long-format data (single observation column with type identifier) via data transformation.

== Creating Population Objects

Pumas requires conversion from tabular data to `Population` objects. We create populations optimized for different analysis types.

=== Population for NLME Modeling

```julia
# Population excluding placebo (for PK modeling)
julia> # population without placebo data
       population = read_pumas(
           (@rsubset wide_data :DOSE ≠ 0);  # exclude placebo subjects
           id=:ID,
           time=:NOMTIME,
           observations=[:CONC, :PD_CONC],
           amt=:AMT,
           cmt=:CMT,
           evid=:EVID,
           covariates=[:WEIGHTB, :SEX, :TRTACT, :DOSE]
       )
Population
  Subjects: 50
  Covariates: WEIGHTB, SEX, TRTACT, DOSE
  Observations: CONC, PD_CONC
```

Key features:
- ```julia (@rsubset wide_data :DOSE ≠ 0)``` excludes placebo subjects (no active drug for PK modeling)
- ```julia observations = [:CONC, :PD_CONC]``` specifies two observation types for multi-endpoint modeling
- ```julia covariates``` declaration enables covariate effects in model specification
- ```julia read_pumas()``` validates data structure: monotonic time, observation-model compatibility, complete covariates

The resulting ```julia Population``` object contains 50 subjects (10 per active dose level) with dosing events and observations.

=== Population Including Placebo

For analyses requiring placebo data (e.g., covariate balance checks, PD baseline assessment):

```julia
julia> # population with placebo data
       population_placebo = read_pumas(
           wide_data;
           id=:ID,
           time=:NOMTIME,
           observations=[:CONC, :PD_CONC],
           amt=:AMT,
           cmt=:CMT,
           evid=:EVID,
           covariates=[:WEIGHTB, :SEX, :TRTACT, :DOSE]
       )
┌ Warning: Your dataset has bolus doses with zero dose amount.
└ @ Pumas none:2327
Population
  Subjects: 60
  Covariates: WEIGHTB, SEX, TRTACT, DOSE
  Observations: CONC, PD_CONC
```

This population includes all 60 subjects. Placebo subjects have zero concentrations (expected) but contribute PD observations
for assessing natural biomarker variability.

*Design Note*: Population objects are vectors of Subjects. Upon construction, Subject input is validated to make sure that all input meets relevant criteria (monotonic time, valid event IDs, etc). Attempting to use an observation name in ```julia @derived``` that was not specified
in ```julia observations = [:CONC, :PD_CONC]``` produces an informative error at model construction, not during estimation hours later.

== Exploratory Data Analysis

Before modeling, we examine data quality and characteristics through summary tables and visualization.

=== Data Quality Overview

```julia
using SummaryTables

# Comprehensive data overview
overview_table(wide_data)
```

```julia overview_table()``` provides:
- Column data types and summary statistics (mean, median, min, max)
- Number of unique values per column
- Inline distribution plots (sparklines)
- Proportion of missing values

This single function reveals data quality issues: unexpected missing data, incorrect types (e.g., numeric coded as string),
or implausible value ranges (negative concentrations, weights outside physiological range).

=== Demographics Table

```julia
# Baseline demographics stratified by sex
table_one(
    unique(wide_data, :ID),  # One row per subject
    [:WEIGHTB => "Baseline Weight [kg]", :SEX => "Sex"],
    groupby = :SEX,
    show_n = true
)
```

Output shows sample sizes, mean ± SD for weight stratified by sex. This confirms covariate balance and identifies potential
confounding (e.g., if all females received low doses).

See Appendix A.2 for extended demographic tables stratified by dose level and additional covariates.

=== Concentration-Time Profile Visualization

Visual exploration reveals dose'response relationships, drug accumulation, and between'subject
variability. See Appendix A.2 for complete concentration-time profile visualization code and additional exploratory analyses.

#figure(
  image("figures/fig2-concvstime.png", width: 100%),
  caption: [Observed PK concentration-time profiles for Days 1 and 6. Individual profiles (gray) and mean profiles by dose level (red) demonstrate dose proportionality, drug accumulation to steady-state (Day 6 concentrations higher than Day 1), and between-subject variability in pharmacokinetics. Log-scale y-axis enables visualization of absorption, distribution, and elimination phases across three orders of magnitude.],
) <fig-conc-time>

This visualization reveals:
- Dose proportionality if any
- Accumulation: Do Day 6 concentrations exceed Day 1 at equivalent times post-dose (half-life > dosing interval)
- Variability: Individual profiles spread around mean (inter-subject variability in clearance, volume, absorption)
- Data quality: Smooth profiles without sudden jumps suggest no major data entry errors

=== PD Data Exploration

Pharmacodynamic endpoints require similar exploration. See Appendix A.2 for:
- PD biomarker time-course stratified by dose
- Exposure-response scatter plots (PD vs PK concentration)
- Hysteresis analysis (temporal relationship between PK and PD)
- Covariate effects on PD baseline

*Key Insight from EDA*: PD response shows counter-clockwise hysteresis (effect continues rising as concentrations fall),
suggesting indirect response mechanism rather than direct effect. This motivates the indirect response model in Section 6.

== Summary: Data Preparation Stage

At this stage we have:
- Loaded and validated data from CSV
- Created ```julia Population``` objects for NLME modeling
- Confirmed data quality through overview tables
- Verified demographic balance across dose groups
- Visualized PK concentration-time profiles revealing dose proportionality and accumulation
- Identified PD characteristics suggesting indirect response dynamics

The ```julia Population``` object is ready for model fitting. Population structure (50 subjects, multi-endpoint observations, covariates)
informs model specification in the next section.

*Appendix References*:
- Appendix A.1: Complete data simulation code
- Appendix A.2: Extended EDA (additional tables, PD visualization, exposure-response analysis)
- Appendix A.3: Non-compartmental analysis workflow

= Model Specification with the ```julia @model``` Macro
<sec-model-spec>

Pumas models are defined using the ```julia @model``` macro, which provides a declarative domain-specific language for pharmacometric model specification.
This section demonstrates model construction through progressive examples: one-compartment, covariate addition, and multi-endpoint PK-PD.

== Declarative Modeling Philosophy

Traditional pharmacometric software uses procedural code where model logic is interleaved with implementation details.
Pumas adopts a declarative approach: specify *what* the model is rather than *how* to compute it.
The ```julia @model``` macro enforces structure through functional blocks, each with a specific purpose.

Benefits of declarative specification:
- *Readability*: Model structure is explicit and matches mathematical notation
- *Composability*: Models are Julia objects that can be stored, modified and version-controlled
- *Safety*: Type system catches structural errors before estimation
- *Flexibility*: Can express diverse models (continuous, discrete, time-to-event, multi-endpoint) with consistent syntax

== One-Compartment Oral Model

We begin with a one-compartment model with first-order absorption—the simplest model capable of describing oral dosing pharmacokinetics.

```julia
oral_model = @model begin
    @metadata begin
        desc = "One-compartment oral absorption model"
        timeu = u"hr"
    end

    @param begin
        θka ∈ RealDomain(lower = 0.1, init = 0.4)
        θcl ∈ RealDomain(lower = 0.1, init = 7.3)
        θvc ∈ RealDomain(lower = 1.0, upper = 500, init = 93.0)

        Ω_pk ∈ PDiagDomain(3)  # 3×3 diagonal covariance

        σ_add_pk ∈ RealDomain(lower = 0.0, init = 1.0)
        σ_prop_pk ∈ RealDomain(lower = 0.0, init = 0.09)
    end

    @random begin
        η ~ MvNormal(Ω_pk)
    end

    @pre begin
        Ka = θka * exp(η[1])
        CL = θcl * exp(η[2])
        Vc = θvc * exp(η[3])
    end

    @dynamics Depots1Central1

    @derived begin
        ipred := @. Central / Vc
        CONC ~ @. Normal(ipred, sqrt(σ_add_pk^2 + (ipred * σ_prop_pk)^2))
    end
end
## output
PumasModel
  Parameters: θka, θcl, θvc, Ω_pk, σ_add_pk, σ_prop_pk
  Random effects: η
  Covariates:
  Dynamical system variables: Depot, Central
  Dynamical system type: Closed form
  Derived: CONC
  Observed: CONC
```

=== Block-by-Block Explanation

*```julia @metadata``` Block*: Documentation and unit specification. ```julia desc``` provides human-readable description. ```julia timeu = u"hr"``` specifies time units (hours),
used in plotting axis labels and reporting. Metadata is optional but recommended for model documentation.

*```julia @param``` Block*: Population parameters with domains and initial values.

Parameter types:
- ```julia RealDomain```: Scalar real parameters with optional bounds. Example: ```julia θka ∈ RealDomain(lower=0.1)``` constrains absorption rate to positive values.
  ```julia init=0.4``` provides reference or starting value for simulation or optimization.
- ```julia PDiagDomain(n)```: Diagonal covariance matrix of dimension n. ```julia Ω_pk ∈ PDiagDomain(3)``` creates 3×3 diagonal matrix for random effects variance.

Domain constraints prevent optimization in invalid parameter space. Attempting ```julia θcl < 0``` (negative clearance) violates ```julia RealDomain(lower=0.1)```,
causing optimizer to reject that parameter value.

*```julia @random``` Block*: Distributions for inter-subject variability.

- ```julia η ~ MvNormal(Ω_pk)``` declares random effects vector η drawn from multivariate normal distribution with zero mean and covariance Ω_pk.
Dimension of η (3 elements) must match dimension of Ω_pk (3×3 matrix).

*```julia @pre``` Block*: Individual parameter derivations.

Individual parameters combine population values (θ) and individual random effects (η):
- ```julia Ka = θka * exp(η[1])```: Log-normal distribution for absorption rate (always positive)
- ```julia CL = θcl * exp(η[2])```: Log-normal clearance
- ```julia Vc = θvc * exp(η[3])```: Log-normal volume

Log-normal parameterization @bonate2011pharmacokinetic @gabrielsson2006pharmacokinetic ensures parameters remain positive and produces right-skewed distributions typical of PK parameters.

*```julia @dynamics``` Block*: System dynamics specification.

```julia Depots1Central1``` invokes a built-in closed-form solution for one-compartment model with first-order absorption.
Pumas recognizes standard compartmental models and uses analytical solutions when available, avoiding numerical ODE solving.

Equivalent differential equation specification (not needed here, but illustrative):
```julia
@dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
end
```

*```julia @derived``` Block*: Observation model with error structure.

- ```julia ipred := @. Central / Vc```: Intermediate variable for individual prediction (concentration = amount / volume). ```julia :=``` informs the program not to output this variable. The variable can used in the rest of the block.
- ```julia CONC ~ @. Normal(ipred, ...)```: CONC observations follow normal distribution with mean ```julia ipred``` and combined additive+proportional error.
  - Error model: $sigma = sqrt(sigma_"add"^2 + (C dot sigma_"prop")^2)$ where C is predicted concentration. This structure allows additive error to dominate at low concentrations and proportional error at high concentrations.
  - ```julia @.``` broadcasts operations element-wise (vectorization). Without ```julia @.```, each scalar operation would need explicit broadcasting.

== Two-Compartment Model with Covariates

Real PK data often exhibit distribution phase requiring two-compartment models. We extend the base model to include peripheral compartment
and covariate effects.

=== Structural Extension: Adding Peripheral Compartment

```julia
oral_model_2cmt = @model begin
    @metadata begin
        desc = "Two-compartment oral model with covariates"
        timeu = u"hr"
    end

    @param begin
        θka ∈ RealDomain(lower = 0.01, init = 1.0)
        θcl ∈ RealDomain(lower = 0.01, init = 1.0)
        θvc ∈ RealDomain(lower = 0.01, upper = 190.0, init = 10.0)
        θq ∈ RealDomain(lower = 0.01, upper = 90.0, init = 10.0)    # Inter-compartmental clearance
        θvp ∈ RealDomain(lower = 0.01, upper = 190.0, init = 10.0)  # Peripheral volume

        # Covariate parameters
        θsexCLF ∈ RealDomain(lower = -10.0, upper = 10.0, init = 1.0)  # Sex effect on CL
        θvcWEIGHTB ∈ RealDomain(lower = 0.1, upper = 10.0, init = 1.0) # Weight exponent on Vc

        Ω_pk ∈ PDiagDomain(4)  # 4 random effects (Ka removed based on diagnostics)

        σ_add_pk ∈ RealDomain(lower = 0.0, init = 1.0)
        σ_prop_pk ∈ RealDomain(lower = 0.0, init = 0.09)
    end

    @covariates SEX WEIGHTB  # Declare covariates used in model

    @random begin
        η ~ MvNormal(Ω_pk)
    end

    @pre begin
        # Categorical covariate: sex effect on clearance
        COVsexCL = SEX == "Female" ? 1 + θsexCLF :
                   (SEX == "Male" ? 1 :
                    error("Expected SEX to be Female or Male, got $SEX"))

        # Continuous covariate: weight effect on volume (power model)
        Ka = θka  # Fixed Ka (no random effect)
        CL = θcl * COVsexCL * exp(η[1])
        Vc = θvc * (WEIGHTB / 77)^θvcWEIGHTB * exp(η[2])
        Q  = θq * exp(η[3])
        Vp = θvp * exp(η[4])
    end

    @dynamics Depots1Central1Periph1  # Two-compartment closed-form solution

    @derived begin
        ipred := @. Central / Vc
        CONC ~ @. Normal(ipred, sqrt(σ_add_pk^2 + (ipred * σ_prop_pk)^2))
    end
end
## output
PumasModel
  Parameters: θka, θcl, θvc, θq, θvp, θsexCLF, θvcWEIGHTB, Ω_pk, σ_add_pk, σ_prop_pk
  Random effects: η
  Covariates: SEX, WEIGHTB
  Dynamical system variables: Depot, Central, Peripheral
  Dynamical system type: Closed form
  Derived: CONC
  Observed: CONC
```

=== Covariate Implementation Patterns

*Categorical Covariates (Sex on Clearance)*:

Ternary operator implements sex effect with error handling:
```julia
COVsexCL = SEX == "Female" ? 1 + θsexCLF :
           (SEX == "Male" ? 1 :
            error("Expected SEX to be Female or Male, got $SEX"))
```

- If `SEX == "Female"`: clearance multiplier = $1 + theta_"sexCLF"$
- If `SEX == "Male"`: clearance multiplier = 1 (reference group)
- Otherwise: error with informative message

This parameterization ensures ```julia θsexCLF = 0``` removes sex effect (facilitates model comparison). ```julia θsexCLF = -0.3``` means female clearance is 70% of male clearance.

Error handling catches data issues early. If data contains ```julia SEX = "F"``` instead of ```julia "Female"```, the model errors immediately during individual parameter calculation,
identifying the specific subject with the issue.

*Continuous Covariates (Weight on Volume)*:

Power model centered at reference value:
```julia
Vc = θvc * (WEIGHTB / 77)^θvcWEIGHTB * exp(η[2])
```

- Reference weight: 77 kg (approximate dataset median)
- At WEIGHTB = 77 kg: $(77 slash 77)^theta = 1$ (no adjustment)
- ```julia θvcWEIGHTB = 0```: removes weight effect ($(W slash 77)^0 = 1$ for all W)
- ```julia θvcWEIGHTB = 1```: allometric scaling (Vc proportional to weight)
- ```julia θclWEIGHTB = 0.75```: typical allometric exponent @anderson2008allometric @west1997allometric for clearance parameters

Standardizing at median weight makes ```julia θvc``` interpretable as typical volume for median-weight subject.

== Multi-Endpoint PK-PD Model

Complex models require explicit ordinary differential equations. We extend to pharmacodynamics using indirect response @dayneka1993indirect @jusko1994indirect with inhibition of degradation (Kout inhibition), effect compartment for hysteresis, and Hill coefficient for steep exposure-response.

```julia
mdl_pkpd = @model begin
    @metadata begin
        desc = "Simultaneous PK-PD model with indirect response (Kout inhibition, effect compartment, Hill coefficient)"
        timeu = u"hr"
    end

    @param begin
        # PK parameters (same as covariate model)
        θka ∈ RealDomain(lower = 0.01)
        θcl ∈ RealDomain(lower = 0.01)
        θvc ∈ RealDomain(lower = 0.01, upper = 190.0)
        θq ∈ RealDomain(lower = 0.01, upper = 90.0)
        θvp ∈ RealDomain(lower = 0.01, upper = 190.0)
        θsexCLF ∈ RealDomain(lower = -10.0, upper = 10.0)
        θvcWEIGHTB ∈ RealDomain(lower = 0.01, upper = 10.0)

        Ω_pk ∈ PDiagDomain(4)
        σ_add_pk ∈ RealDomain(lower = 0.0)
        σ_prop_pk ∈ RealDomain(lower = 0.0)

        # PD parameters - Kout inhibition with effect compartment and Hill coefficient
        tvkin ∈ RealDomain(lower = 0.001, upper = 30.0, init = 1.5)    # Production rate
        tvkout ∈ RealDomain(lower = 0.001, upper = 100.0, init = 0.03) # Degradation rate
        tvic50 ∈ RealDomain(lower = 0.1, upper = 100.0, init = 12.0)   # Half-maximal inhibitory concentration
        tvimax ∈ RealDomain(lower = 0.0, upper = 1.0, init = 0.9)      # Maximal inhibition
        tvke0 ∈ RealDomain(lower = 0.001, upper = 10.0, init = 0.08)   # Effect compartment rate constant
        tvgamma ∈ RealDomain(lower = 0.1, upper = 10.0, init = 1.5)    # Hill coefficient

        Ωpd ∈ PDiagDomain(2)  # PD random effects (IC50 and Ke0 variability)
        σ_add_pd ∈ RealDomain(lower = 0.0)
        σ_prop_pd ∈ RealDomain(lower = 0.0)
    end

    @covariates SEX WEIGHTB

    @random begin
        η ~ MvNormal(Ω_pk)      # PK random effects
        ηpd ~ MvNormal(Ωpd)     # PD random effects (independent from PK)
    end

    @pre begin
        # PK parameters with covariates (same as before)
        Ka = θka
        COVsexCL = SEX == "Female" ? 1 + θsexCLF :
                   (SEX == "Male" ? 1 :
                    error("Expected SEX to be Female or Male, got $SEX"))
        CL = θcl * COVsexCL * exp(η[1])
        Vc = θvc * (WEIGHTB / 77)^θvcWEIGHTB * exp(η[2])
        Q = θq * exp(η[3])
        Vp = θvp * exp(η[4])

        # PD parameters - Kout inhibition with effect compartment
        kin = tvkin
        kout = tvkout
        ic50 = tvic50 * exp(ηpd[1])  # Inter-subject variability in IC50
        imax = tvimax
        Ke0 = tvke0 * exp(ηpd[2])    # Inter-subject variability in Ke0
        gamma = tvgamma
    end

    @vars begin
        Conc = Central / Vc
        Ce_safe = max(Ce, 0.0)  # Ensure non-negative for numerical stability
        INH = imax * Ce_safe^gamma / (ic50^gamma + Ce_safe^gamma)  # Hill function
    end

    @init begin
        Response = kin / kout  # Baseline steady-state response (~50 units)
        Ce = 0.0               # Effect compartment starts at zero
    end

    @dynamics begin
        Depot' = -Ka * Depot
        Central' = Ka * Depot - (CL + Q) / Vc * Central + Q / Vp * Peripheral
        Peripheral' = Q / Vc * Central - Q / Vp * Peripheral
        Ce' = Ke0 * (Conc - Ce)  # Effect compartment for hysteresis
        Response' = kin - kout * (1 - INH) * Response  # Kout inhibition
    end

    @derived begin
        conc_model := @. abs(Central / Vc)
        CONC ~ @. Normal(conc_model, sqrt(σ_add_pk^2 + (conc_model * σ_prop_pk)^2))
        PD_CONC ~ @. Normal(Response, sqrt(σ_add_pd^2 + (Response * σ_prop_pd)^2))
    end
end
## output
PumasModel
  Parameters: θka, θcl, θvc, θq, θvp, θsexCLF, θvcWEIGHTB, Ω_pk, σ_add_pk, σ_prop_pk, tvkin, tvkout, tvic50, tvimax, tvke0, tvgamma, Ωpd, σ_add_pd, σ_prop_pd
  Random effects: η, ηpd
  Covariates: SEX, WEIGHTB
  Dynamical system variables: Depot, Central, Peripheral, Ce, Response
  Dynamical system type: Nonlinear ODE
  Derived: CONC, PD_CONC, Conc, Ce_safe, INH
  Observed: CONC, PD_CONC, Conc, Ce_safe, INH
```

=== Advanced Block Features

*```julia @vars``` Block*: Defines intermediate variables for code clarity and numerical safety. ```julia Conc = Central / Vc``` computes concentration, ```julia Ce_safe = max(Ce, 0.0)``` ensures non-negative effect compartment values for numerical stability, and ```julia INH``` calculates the Hill function for Kout inhibition.

*```julia @init``` Block*: Initial conditions for differential equations. ```julia Response = kin / kout``` sets baseline PD response to steady-state (~50 units with kin=1.5 and kout=0.03). ```julia Ce = 0.0``` initializes the effect compartment at zero (no drug present at time 0).

*```julia @dynamics``` Block with Explicit ODEs*: Five coupled differential equations:
1. ```julia Depot'```: First-order absorption from depot compartment
2. ```julia Central'```: Drug entering from depot, clearing from central, distributing to peripheral
3. ```julia Peripheral'```: Drug distributing to/from central compartment
4. ```julia Ce'```: Effect compartment equilibrates with central concentration (creates hysteresis)
5. ```julia Response'```: Indirect response with Kout inhibition driven by effect compartment

PD equation breakdown:
- Effect compartment: ```julia Ce' = Ke0 * (Conc - Ce)``` introduces temporal delay between plasma concentration and effect site, producing the counter-clockwise hysteresis observed in EDA
- Inhibition term: $"INH" = "imax" dot "Ce"^gamma slash ("ic50"^gamma + "Ce"^gamma)$ is the Hill function with coefficient γ controlling exposure-response steepness
- Kout inhibition: ```julia Response' = kin - kout * (1 - INH) * Response``` reduces degradation rate when drug is present

When Ce = 0 (no drug), INH = 0, so degradation = kout·Response, yielding steady-state Response = kin/kout ≈ 50.
When Ce >> ic50 (high effect site concentration), INH approaches imax (0.9), reducing effective kout to 0.1×kout, causing Response to rise toward kin/(0.1×kout) ≈ 500.

*```julia @derived``` with Multiple Observations*: Two observation types, each with its own error model:
- ```julia CONC```: PK observations (drug concentration)
- ```julia PD_CONC```: PD observations (biomarker response)

Pumas matches observation names to columns in ```julia Population``` object. If data contains observations for both CONC and PD_CONC at certain times,
both are used in likelihood calculation. If only one is present (e.g., sparse PD sampling), only that observation contributes.

== Design Benefits

*Composability*: Models are Julia objects stored in variables. Can save: ```julia serialize("model.jls", mdl_pkpd)``` and load later.

*Safety*: Type system enforces consistency:
- If ```julia Ω_pk ∈ PDiagDomain(3)``` but ```julia @random``` declares ```julia η ~ MvNormal(Ω_pk)``` and ```julia @pre``` uses ```julia exp(η[4])```, error occurs at model construction:
  "Index 4 out of bounds for 3-element vector"
- If ```julia @derived``` references observation name not in ```julia Population```: "Observation CONCENTRATION not found. Available: [:CONC, :PD_CONC]"

*Clarity*: Block structure makes assumptions explicit. Reviewer sees covariate effects in ```julia @pre```, random effect structure in ```julia @random```,
error model in ```julia @derived```. No need to trace procedural code to infer model structure.

*Extensibility*: Can define custom dynamics using any Julia-compatible ODE solver, custom observation distributions (e.g., BoundedInteger Models),
or user-defined functions for complex PK processes.

*Appendix Reference*: See Appendix A.4 for complete model definitions, including intermediate models (two-compartment without covariates,
models with full vs diagonal covariance), and Appendix A.5 for initial validation procedures.

= Model Fitting and Diagnostics
<sec-fitting>

With models defined, we proceed to parameter estimation and model evaluation. This section demonstrates progressive estimation strategies,
comprehensive diagnostics, model refinement, and uncertainty quantification.

== Progressive Estimation Strategy

Parameter estimation benefits from staged approaches that build from simpler to more complex models.

=== Initial Parameter Validation

Before estimation, verify that initial parameters produce finite likelihood:

```julia
initial_est_oral_model = (
    θka = 0.9,
    θcl = 4.3,
    θvc = 252.0,
    Ω_pk = Diagonal([0.1, 0.1, 0.1]),
    σ_add_pk = sqrt(0.09),
    σ_prop_pk = sqrt(0.01),
)

# Check log-likelihood at initial parameters
ll = loglikelihood(oral_model, population, initial_est_oral_model, FOCE())
```

Finite log-likelihood confirms model structure is correct (no numerical issues, dimensions match). Infinite or NaN likelihood indicates
specification errors (incompatible observation/prediction dimensions, invalid parameter values). To identify individual subjects contributing to NaN likelihood, the ```julia findinfluential()``` function can be used.

=== Warm Start with ```julia NaivePooled```

Fit fixed effects using ```julia NaivePooled``` (assumes no inter-subject variability, Ω = 0):

```julia
np_fit = fit(
    oral_model,
    population,
    (; initial_est_oral_model..., Ω_pk = Diagonal([0.0, 0.0, 0.0])),
    NaivePooled();
    constantcoef = (:Ω_pk,),  # Hold Ω at zero
    optim_options = (; show_trace = false),
)
```

```julia NaivePooled``` estimation is fast (no random effects integration) and provides stable fixed effect estimates.
These estimates initialize the full population model.

=== Population Model with ```julia FOCE```

```julia
foce_fit_2cmt = fit(
    oral_model,
    population,
    initial_est_oral_model,
    FOCE();
    optim_options = (; show_trace = false),
)
```

```julia FOCE``` (First Order Conditional Estimation) is the most commonly used algorithm @bauer2019nonmem @upton2014nonmem for population PK/PD. Estimation time depends on model complexity
and data size. For our case study (two-compartment, 50 subjects, ~1000 observations), ```julia FOCE``` completes in minutes.

See Section 5.6 for formal uncertainty quantification via ```julia infer()```.

== Inspection Framework

The ```julia inspect()``` function generates all diagnostic quantities:

```julia
foce_inspect_2cmt = inspect(foce_fit_2cmt)
```

This creates a ```julia FittedPumasModel``` containing:
- *PRED*: Population predictions (using θ only, no random effects)
- *IPRED*: Individual predictions (using θ and empirical Bayes estimates η̂)
- *CWRES* @hooker2007cwres: Conditional weighted residuals (accounts for uncertainty in η̂)
- *EBEs*: Empirical Bayes estimates of individual random effects (η̂ values)

All outputs are DataFrames accessible via ```julia DataFrame(foce_inspect_2cmt)```, enabling custom analyses.

== Standard Diagnostic Plots

=== Goodness-of-Fit Panel

Four-panel diagnostic plot:

```julia
goodness_of_fit(foce_inspect_2cmt;
    observations = [:CONC],
    markercolor = (:grey, 0.5),
    legend = (position = :bottom, framevisible = false),
    figure = (size = (800, 800), fontsize = 18)
)
```

#figure(
  image("figures/fig3-std2cmpt-gof.png", width: 100%),
  caption: [Goodness-of-fit diagnostics for two-compartment model. Panel 1 (top left): Observations vs population predictions (PRED) show scatter around identity line indicating adequate population-level fit. Panel 2 (top right): Observations vs individual predictions (IPRED) demonstrate tighter scatter, confirming random effects capture between-subject variability. Panel 3 (bottom left): Conditional weighted residuals (CWRES) vs PRED show random scatter around zero without funnel pattern, indicating appropriate error model. Panel 4 (bottom right): CWRES vs time show no systematic trends, confirming structural model adequately captures time-dependent processes (absorption, distribution, elimination).],
) <fig-gof>

Systematic deviations suggest model refinement needed (additional compartments, different error model, covariate effects).

=== Individual Subject Fits

```julia
subject_fits(foce_inspect_2cmt;
    observations = [:CONC],
    paginate = true,
    separate = true,
    figure = (size = (1200, 900), fontsize = 18)
)
```

Each panel shows one subject's observed data (points) with individual prediction curves (lines using EBEs).
```julia paginate = true``` creates multiple figure pages for large datasets (>9 subjects per page).

Useful for identifying:
- Outlier subjects with poor fits
- Subjects driving parameter estimates
- Data entry errors (sudden concentration jumps)

== Visual Predictive Check

VPC assesses whether model generates data consistent with observations.

```julia
# Generate VPC with 200 simulation replicates
foce_vpc_2cmt = vpc(foce_fit_2cmt;
    covariates = [:tad],  # Stratify by time after dose
    ensemblealg = EnsembleThreads()  # Parallel simulation
)

# Plot VPC
vpc_plot(foce_vpc_2cmt;
    observations = true,
    markercolor = :grey,
    observed_linewidth = 4,
    figurelegend = (position = :b, orientation = :horizontal),
    axis = (
        yscale = Makie.pseudolog10,
        ylabel = "Concentration (ng/mL)",
        xlabel = "Time (hours)",
    ),
    figure = (size = (1800, 1400), fontsize = 40)
)
```

#figure(
  image("figures/fig4-std2cmpt-vpc.png", width: 100%),
  caption: [Visual predictive check for two-compartment model. Observed data 5th, 50th, and 95th percentiles (lines with points) fall within simulated prediction intervals (shaded regions), confirming model adequately captures both central tendency and variability across the concentration-time profile. Parallel simulation with 200 replicates completed in minutes using automatic multithreading.],
) <fig-vpc>


VPC interpretation:
- Observed percentiles within prediction intervals → model captures variability correctly
- Observed median outside interval → systematic bias in central tendency
- Observed extremes (5th/95th) outside intervals → random effect variance misspecified

VPC stratification by covariates (dose, weight, sex) checks whether model captures covariate effects appropriately.

== Model Comparison and Refinement

=== Structural Comparison

Compare one-compartment vs two-compartment models:

```julia
# Parameter estimates side-by-side
compare_estimates(one_compartment = foce_fit, two_compartment = foce_fit_2cmt)

# Information criteria
using DataFramesMeta

comparison = @chain begin
    leftjoin(
        metrics_table(foce_fit),
        metrics_table(foce_fit_2cmt),
        on = :Metric,
        makeunique = true
    )
    rename(:Value => :One_Cmt, :Value_1 => :Two_Cmt)
end
```

Metrics include:
- *Log-likelihood*: Higher is better (less penalized for complexity)
- *AIC* @akaike1974aic: $-2 dot "LL" + 2 dot k$ where k = number of parameters. Lower is better.
- *BIC* @schwarz1978bic: $-2 dot "LL" + k dot log(n)$ where n = number of observations. Lower is better, penalizes complexity more than AIC.

Two-compartment model has lower BIC despite additional parameters (Q, Vp), indicating improved fit outweighs complexity penalty.

=== Random Effect Refinement

Diagnostic plots may reveal random effects with low variability (near zero variance) or high shrinkage (EBEs pulled toward zero).
Consider removing such random effects:

```julia
# Two-compartment without Ka random effect
oral_model_2cmt_noka = @model begin
    # ... (similar to oral_model_2cmt but Ω_pk ∈ PDiagDomain(4) instead of 5)
    @pre begin
        Ka = θka  # Fixed, no exp(η[...])
        CL = θcl * exp(η[1])
        Vc = θvc * exp(η[2])
        Q = θq * exp(η[3])
        Vp = θvp * exp(η[4])
    end
    # ...
end

foce_noka = fit(
    oral_model_2cmt_noka,
    population,
    (; coef(foce_fit_2cmt)..., Ω_pk = Diagonal([0.1, 0.1, 0.1, 0.1])),
    FOCE()
)

# Compare BIC
bic(foce_fit_2cmt), bic(foce_noka)  # Lower BIC favors simpler model
```

Full code for model without Ka random effect in Appendix A.7.

=== Covariate Identification

Plot empirical Bayes estimates against covariates to identify relationships:

```julia
empirical_bayes_vs_covariates(foce_inspect_2cmt_noka;
    covariates = [:WEIGHTB, :SEX, :DOSE],
    categorical = [:SEX, :DOSE],
    color = (:grey, 0.5)
)
```

Interpretation:
- *Trend in η[Vc] vs WEIGHTB*: Positive correlation suggests weight effect on Vc
- *Different η[CL] distributions for males vs females*: Suggests sex effect on CL
- *Random scatter*: No covariate effect worth modeling

After identifying covariates, implement in model (as shown in Section 4) and re-fit. EBE plots for covariate model should show reduced trends,
confirming covariates explain variability previously captured by random effects.

See Appendix A.9 for complete covariate modeling workflow including automated selection via ```julia covariate_select()```.

== Parameter Uncertainty Quantification

After finalizing model structure, quantify parameter uncertainty:

```julia
infer_result = infer(foce_fit_covariate; method = :sandwich, level = 0.95)
coeftable(infer_result)
```

Output table includes:
- *Estimate*: Point estimates (θ̂, Ω̂, σ̂)
- *SE*: Standard errors from sandwich estimator (observed Fisher information)
- *95% CI*: Confidence intervals (Estimate ± 1.96 × SE for normal approximation)
- *RSE%*: Relative standard error (SE / Estimate × 100%). RSE < 30% indicates good precision.

Alternative inference methods:

```julia
# Bootstrap (non-parametric, computationally intensive)
infer_bootstrap = infer(foce_fit_covariate; method = :bootstrap, samples = 200)

# Sampling-importance-resampling (SIR, semi-parametric)
infer_sir = infer(foce_fit_covariate; method = :sir, samples = 1000)
```

Bootstrap resamples subjects with replacement, re-estimates parameters for each resample. Provides non-asymptotic confidence intervals
but requires substantial computation (200 resamples × model fit time).

SIR samples from asymptotic distribution, weights by likelihood ratios. Faster than bootstrap, more robust than sandwich for small samples.

See Appendix A.10 for detailed inference method comparison and variance-covariance matrix interpretation.

== Summary: Model Fitting and Diagnostics

Progressive workflow:
1. Validate initial parameters with ```julia loglikelihood()```
2. Warm start with ```julia NaivePooled()``` for fixed effects
3. Estimate population model with ```julia FOCE()```
4. Generate diagnostics with ```julia inspect()``` and standard plotting functions
5. Assess model adequacy via GOF plots and VPC
6. Compare competing models using AIC/BIC
7. Refine random effect structure based on shrinkage and variance estimates
8. Identify and implement covariate effects using EBE diagnostics
9. Quantify final parameter uncertainty with ```julia infer()```

The integrated workflow enables rapid iteration. Diagnostic plots generate with single function calls. VPC simulation exploits parallel computing.
Model comparison uses consistent interfaces.

For our case study, the final model is two-compartment with sex on CL, weight on Vc, and four random effects (CL, Vc, Q, Vp).
This model serves as the PK component for multi-endpoint PK-PD analysis in Section 6.

*Appendix References*:
- Appendix A.5: Initial validation and prior predictive checks
- Appendix A.6: Complete fitting workflows with convergence diagnostics
- Appendix A.7: Extended diagnostics (shrinkage, influence, additional plots)
- Appendix A.8: Model comparison procedures and likelihood ratio testing
- Appendix A.9: Covariate modeling details and automated selection
- Appendix A.10: Inference methods (sandwich, bootstrap, SIR) and interpretation

= Multi-Endpoint PK-PD Modeling
<sec-pkpd>

With an adequate PK model established, we extend to pharmacodynamics. This section demonstrates multi-endpoint modeling using
the PK-PD model defined in Section 4.3, focusing on the workflow from exposure-response exploration through parameter estimation.

== Exposure-Response Analysis

Before fitting the PK-PD model, exposure-response visualization reveals the underlying concentration-effect relationship.
Using DataFramesMeta.jl, we extract matched PK-PD observations from the inspection results and bin concentrations
to calculate mean PD responses with standard errors across concentration ranges.

#figure(
  image("figures/fig5-errelation.png", width: 100%),
  caption: [Exposure-response relationship showing PD biomarker response vs PK concentration. Individual observations (points colored by dose) and binned means ± standard error (black squares with whiskers) demonstrate saturable response consistent with Kout inhibition model with Hill coefficient. Counter-clockwise hysteresis (response rising as concentrations fall) confirms effect compartment mechanism. Log-scale x-axis spans physiological concentration range.],
) <fig-exposure-response>

The exposure-response plot (Figure 5) reveals characteristic sigmoidal kinetics with hysteresis: PD response increases nonlinearly with effect site concentration,
approaching a plateau consistent with the Hill function $"imax" dot "Ce"^gamma slash ("ic50"^gamma + "Ce"^gamma)$ in the model.
Individual variability around binned means reflects inter-subject differences in IC50 and Ke0, captured by the random effect structure. The counter-clockwise hysteresis (response continues rising as plasma concentrations fall) confirms the effect compartment mechanism.
This visualization confirms the indirect response model with Kout inhibition and effect compartment is mechanistically appropriate.

== Simultaneous PK-PD Estimation

The PK-PD model (Section 4.3) handles both CONC and PD_CONC observations within a single ```julia @derived``` block.
Pumas automatically matches observation names to data columns, calculating likelihood contributions from
whichever observations are present at each time point. This design enables sparse PD sampling schedules
without special handling—missing observations simply contribute nothing to the likelihood at those times.

=== Parameter Initialization Strategy

PD parameter initialization benefits from physiological reasoning. The production rate (kin = 1.5) and degradation rate (kout = 0.03 hr⁻¹, reflecting ~23-hour biomarker half-life) combine to yield baseline response of ~50 units. The half-maximal inhibitory concentration (IC50 = 12) is set within the observed effect site concentration range. Maximal inhibition (Imax = 0.9) represents 90% reduction in degradation at saturating concentrations. The effect compartment rate constant (Ke0 = 0.08 hr⁻¹) controls hysteresis magnitude—slower Ke0 produces more pronounced delay between plasma concentration and effect. The Hill coefficient (γ = 1.5) determines exposure-response steepness.

Pumas enables parameter reuse across models through Julia's splat operator: ```julia (; coef(pk_fit)..., new_pd_params...)```
unpacks all PK estimates and appends PD parameters, ensuring consistency without manual transcription.

=== Fixing Well-Characterized Parameters

When PK parameters are well-estimated from rich sampling, fixing them during PD estimation improves stability:

```julia
pd_fit = fit(
    mdl_pkpd,
    population,
    init_θ_pkpd,
    FOCE();
    constantcoef = keys(coef(pk_fit)),  # Fix all PK parameters
)
```

The ```julia constantcoef``` argument accepts parameter names to hold constant. Here, all PK parameters remain at their
previously estimated values while only PD parameters (kin, kout, IC50, Imax, Ke0, gamma, and associated variability terms) are optimized.

This approach offers practical benefits: faster estimation from reduced parameter space, stable convergence
since PK dynamics are already validated, and tractability even with sparse PD data. The alternative—removing
```julia constantcoef``` to re-estimate everything simultaneously—accounts for PK-PD parameter correlations but requires
richer data and longer computation times.

=== Diagnostics for Multi-Endpoint Models

The same diagnostic functions apply to PD endpoints. Calling ```julia inspect()``` on the fitted PK-PD model generates
predictions and residuals for both observation types. Diagnostic plots accept an ```julia observations``` argument
to focus on specific endpoints:

```julia
goodness_of_fit(inspect(pd_fit); observations = [:PD_CONC])
```

For PD with indirect response dynamics, particular attention goes to time trends in residuals—systematic
patterns would indicate the model fails to capture the temporal delay between concentration changes and
response changes. Random CWRES scatter around zero confirms the indirect response mechanism adequately
describes the observed hysteresis.

== Sequential vs Simultaneous Modeling

An alternative workflow, sequential modeling, treats individual PK parameters as fixed inputs for the PD model. In Pumas, this is implemented by passing the full set of estimated PK coefficients to the ```julia fit``` function via the ```julia constantcoef``` argument, ensuring that only PD-specific parameters are optimized while maintaining the PK structure. This approach suits scenarios where PK and PD data come from different studies or where separate teams develop each model component.

Sequential modeling trades statistical efficiency for computational convenience. It does not propagate PK parameter uncertainty into PD estimates and cannot capture PK-PD parameter correlations. For our case study with rich PK and PD data from the same subjects, simultaneous modeling (with fixed PK parameters) provides appropriate uncertainty characterization. Appendix A.11 details both workflows with complete implementations.

== Design Philosophy: Unified Multi-Endpoint Handling

Pumas treats multi-endpoint models as a natural extension of single-endpoint analysis rather than a special case
requiring different syntax or workflows. The same ```julia fit()``` function handles one or multiple observation types.
The same ```julia inspect()``` framework produces diagnostics for all endpoints. The same ```julia simobs()``` generates predictions
across all model outputs.

This uniformity reflects a core design principle: complexity in the scientific problem (multiple endpoints,
indirect mechanisms, covariate effects) should not require complexity in the software interface.
The ```julia @model``` macro's block structure isolates each modeling concern—parameters, dynamics, observations—so
extending from PK-only to PK-PD requires adding blocks rather than learning new APIs.

== Summary

Multi-endpoint PK-PD modeling in Pumas follows the same workflow as single-endpoint analysis:
define the model with ```julia @model```, validate initial parameters, fit with ```julia fit()```, diagnose with ```julia inspect()```.
The exposure-response relationship with hysteresis confirms Kout inhibition with effect compartment is appropriate for indirect response modeling.
Fixing well-characterized PK parameters during PD estimation balances statistical rigor with computational efficiency.
The validated PK-PD model now enables simulation-based decision support in Section 7.

*Appendix Reference*: Appendix A.11 provides complete code for simultaneous and sequential PK-PD workflows.

= Simulation for Decision Support
<sec-simulation>

The validated PK-PD model enables simulation of unobserved scenarios—doses, durations, and populations beyond those studied.
This section demonstrates dose selection through target attainment analysis, translating model predictions into
clinically interpretable metrics that inform development decisions.

== Clinical Question and Decision Framework

*Objective*: Select optimal dose for 14-day treatment based on PD biomarker target attainment.

*Therapeutic Range*: 75-125 for PD biomarker (defined by efficacy/safety considerations from prior studies).

*Candidate Regimens*: 100, 400, 800, or 1600 mg once daily for 14 days.

*Decision Metrics*:
1. *Probability of Target Attainment (TA)*: Proportion of subjects achieving PD response within therapeutic range at any time during treatment
2. *Time to Target (TTA)*: Median hours to first observation within therapeutic range
3. *Time in Therapeutic Range (TTR)*: Percentage of steady-state dosing interval (Day 14) spent within range

These metrics capture distinct clinical considerations: TA addresses whether the dose achieves therapeutic effect at all,
TTA quantifies onset rapidity, and TTR measures maintenance of therapeutic levels—relevant when sustained biomarker
control correlates with clinical outcomes.

== Virtual Population and Dosing Regimen Specification

Clinical trial simulation requires generating virtual subjects with realistic covariate distributions.
Pumas provides two constructs for this purpose: the ```julia Subject``` type for individual patients and ```julia DosageRegimen```
for specifying dosing schedules.

Virtual subjects are created by sampling covariates from distributions matching the target population.
Sex is sampled with equal probability; weight follows sex-specific normal distributions derived from
observed study data (females: mean 75 kg, SD 3 kg; males: mean 85 kg, SD 4 kg). Each subject receives
a dosing regimen specified through ```julia DosageRegimen```:

```julia
DosageRegimen(dose; addl = 13, ii = 24)  # 14 days QD dosing
```

The ```julia addl``` (additional doses) and ```julia ii``` (interdose interval) arguments encode complex schedules concisely:
an initial dose plus 13 additional doses at 24-hour intervals yields 14 total doses. This same syntax
handles loading doses, split dosing, or any schedule representable as repeated events at fixed intervals.

For the current analysis, 100 virtual subjects per dose level (400 total) provide adequate precision
for target attainment probability estimates. Using StableRNGs.jl ensures reproducibility—the same
random seed generates identical virtual populations across analysis runs.

== Simulation Execution and Output Handling

The ```julia simobs()``` function executes simulation using the validated PK-PD model and estimated parameters:

```julia
sim_results = simobs(mdl_pkpd, pop_scenarios, coef(pd_fit);
    obstimes = 0:0.5:336,       # Every 0.5 hours for 14 days
    simulate_error = false)     # Exclude residual variability
```

Two options merit attention. Setting ```julia obstimes``` to a fine time grid (0.5-hour resolution) enables accurate
calculate time-based metrics like TTR. Setting ```julia simulate_error = false``` generates predictions conditional
on random effects without adding residual variability—isolating biological variability (inter-subject differences) from
measurement noise (assay error), which is appropriate for projecting target attainment in future populations.

Simulation output converts directly to DataFrame for analysis using standard Julia data manipulation tools.
This interoperability eliminates format conversion steps between simulation and metric calculation.

== Target Attainment Analysis

From the simulated PD time courses, per-subject metrics are calculated: whether the subject ever reached
therapeutic range (TA), when they first reached it (TTA), and what fraction of the steady-state dosing
interval they spent within range (TTR). These individual metrics aggregate to population-level summaries
with confidence intervals reflecting simulation uncertainty.

#figure(
  image("figures/fig6-pta.png", width: 100%),
  caption: [Simulated steady-state biomarker profiles at dose 6 (Day 14) across six dosing regimens (0–1600 mg). Lines indicate median predictions; shaded bands
    represent 90% prediction intervals from 100 simulated trials. Dashed horizontal lines denote the therapeutic range (75–125). Higher doses produce greater
    biomarker elevation and prolonged time within the therapeutic window.],
) <fig-pta>

Figure 6 displays simulated steady-state PD profiles for each dose level, showing median response with 90% prediction
intervals across the virtual population. The therapeutic range boundaries (75-125) overlay the profiles,
enabling visual assessment of target attainment and excursion patterns.

// Artifact 7: Table 1 - Target Attainment Metrics by Dose

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: none,

    // Header
    table.hline(),
    [], table.cell(colspan: 6)[*Dose, mg*],
    [], [0], [100], [200], [400], [800], [1600],
    table.hline(),

    // Probability of TA
    table.cell(colspan: 7)[*Probability of TA*],
    [#h(1em) Median], [0], [40], [100], [100], [100], [100],
    [#h(1em) 95% CI], [\[0, 0\]], [\[14.8, 60\]], [\[90, 100\]], [\[100, 100\]], [\[100, 100\]], [\[100, 100\]],

    // Time to Target
    table.cell(colspan: 7)[*Time to Target*],
    [#h(1em) Median], [--], [124], [120], [120], [120], [120],
    [#h(1em) 95% CI], [--], [\[120, 133\]], [\[120, 121\]], [\[120, 120\]], [\[120, 120\]], [\[120, 120\]],

    // Time in TR
    table.cell(colspan: 7)[*Time in TR*],
    [#h(1em) Median], [--], [82.8], [99.2], [99.2], [99.2], [99.2],
    [#h(1em) 95% CI], [--], [\[46.7, 99.2\]], [\[93, 99.2\]], [\[99.2, 99.2\]], [\[99.2, 99.2\]], [\[99.2, 99.2\]],

    table.hline(),
  ),
  caption: [
    Target attainment metrics by dose level summarized across 100 simulated trials.
    Median and 95% confidence intervals shown for: probability of target attainment
    (TA, percent of subjects reaching therapeutic range), time to target (TTA, hours
    to first therapeutic observation), and time in therapeutic range (TTR, percent of
    dosing interval). Higher doses increase TA and TTR but may risk exceeding upper
    therapeutic threshold. Optimal dose balances efficacy and safety considerations.
  ],
) <tbl-target-attainment>

== Dose Selection Rationale

The metrics reveal distinct dose-response patterns. The 100 mg dose achieves target in only 40% of subjects with prolonged onset (124 hours) and limited time in range (82.8%), indicating subtherapeutic exposure. The 200 mg dose reaches near-complete target attainment (100%) with consistent onset (120 hours) and high time in range (99.2%).

Higher doses (400-1600 mg) maintain 100% target attainment with identical onset times (120 hours) and time in range (99.2%), suggesting a plateau effect beyond 200 mg. Figure 6 shows median responses within therapeutic range across all active doses, with 90% prediction intervals narrowing at higher doses.

*Recommendation*: 200 mg QD provides optimal risk-benefit profile—complete target attainment with minimal dose required, reducing potential for exposure-related adverse effects while ensuring therapeutic efficacy for Phase 2 development.

== Design Philosophy: Unified Simulation Framework

The same ```julia simobs()``` function serves multiple purposes throughout the workflow:
- Prior predictive checks (Section 4): Assessing whether initial parameters generate plausible data
- Visual predictive checks (Section 5): Validating model fit against observed data
- Scenario evaluation (this section): Comparing unobserved dosing regimens

No separate syntax or specialized functions for different simulation contexts. The model object, population
structure, and parameter interface remain consistent. What changes is the input—initial guesses, fitted
estimates, or hypothetical values—and the downstream analysis applied to simulation output.

This uniformity reflects a design principle: the mathematical operation of simulating from a model is
identical regardless of purpose. Separating simulation mechanics from application-specific analysis
enables flexible reuse while maintaining a minimal, learnable API.

== Summary

Model-based simulation enables evaluation of scenarios not observed in clinical trials:
- Doses beyond the studied range (100 mg and 1600 mg extrapolated from 200-800 mg observations)
- Extended treatment duration (14 days simulated from 6-day study data)
- Decision metrics (time in range, time to target) not prospectively defined in the original protocol

Quantitative comparison of dosing regimens provides evidence for dose selection without additional clinical studies.
Target attainment analysis translates model predictions into clinically interpretable metrics aligned with
regulatory expectations for exposure-response characterization and probability of efficacy.

The workflow demonstrated—model development → validation → simulation → decision metrics—exemplifies
model-informed drug development @marshall2016midd @barrett2022midd (MIDD), where quantitative modeling
directly supports clinical and regulatory decision-making.

*Appendix Reference*: Appendix A.12 provides complete metric calculation code, extended scenarios
(alternative dosing frequencies, population subgroups), sensitivity analyses to parameter uncertainty,
and guidelines for simulation study design.

= Discussion
<sec-discussion>

== Platform Integration: Realized Benefits

Pumas integrates the pharmacometric workflow within a single computational framework. This article demonstrated the integration
through a multiple ascending dose study analyzed from data preparation through model-based dose selection.

=== Single-Environment Workflow

The entire analysis executed in Julia without tool switching:
- Data manipulation (DataFramesMeta.jl)
- Population object creation (```julia read_pumas()```)
- Model specification (```julia @model``` macro)
- Parameter estimation (FOCE, NaivePooled)
- Diagnostics (inspect framework, GOF plots, VPC)
- Simulation (```julia simobs()``` for scenarios)
- Reporting (SummaryTables.jl, AlgebraOfGraphics.jl)

Eliminating tool boundaries removes data format conversion steps. The same ```julia Population``` object serves model fitting, and simulation.
The same ```julia @model``` specification handles validation, estimation, and scenario evaluation. Intermediate results remain in memory as Julia objects,
enabling rapid iterative refinement without file I/O overhead.

=== Input Validation and Error Detection

Type constructors catch errors at construction:
- ```julia read_pumas()``` validates monotonic time, observation-model matching, covariate completeness—errors report specific subjects and times requiring correction
- ```julia @model``` macro checks parameter domain consistency, random effect dimensions, observation name matching—errors occur at model construction (seconds),
  not during overnight estimation runs

Informative error messages reduce debugging time. Rather than cryptic numerical failures during optimization, users receive actionable guidance:
"Observation CONCENTRATION not found. Available: [:CONC, :PD_CONC]" directs immediate correction.

=== API Consistency Through Multiple Dispatch

Function interfaces follow uniform patterns:
- ```julia fit(model, population, params, algorithm)``` handles all estimation methods with the same signature
- ```julia simobs(model, population, params; options...)``` works for prior checks, VPCs, and scenario evaluation
- ```julia inspect(fitted_model)``` produces diagnostic quantities regardless of model complexity
- ```julia goodness_of_fit()``` and other diagnostic plots accept an ```julia observations``` argument to focus on specific endpoints

Consistency reduces cognitive load. Learning one workflow (e.g., ```julia FOCE``` estimation for a PK model) transfers directly to other contexts
(```julia FOCE``` for PK-PD, ```julia SAEM``` for complex random effects, Bayesian for prior incorporation). As demonstrated in Sections 6 and 7,
extending from single-endpoint PK to multi-endpoint PK-PD and from model validation to scenario simulation requires no new syntax—only
different inputs to the same functions.

=== Reproducibility Infrastructure

Version control and environment management ensure reproducibility:
- *Project.toml*: Lists package dependencies with version constraints
- *Manifest.toml*: Locks exact versions of all packages and dependencies
- *Quarto integration*: Combines code, results, and narrative in single document
- *Reproducible RNGs*: `StableRNG(seed)` ensures identical simulation results across runs

Sharing analysis code with Project.toml/Manifest.toml enables collaborators to reconstruct identical computational environments.
Quarto notebooks execute code blocks, capture outputs, and render to publication-quality documents with single command (`quarto render`).

== Current Platform Scope and Capabilities

=== Supported Applications

Pumas supports:
- *Phase 1-3 clinical trials*: PK, PK-PD, dose-response modeling for regulatory submissions @fda2022popPK
- *Non-compartmental analysis*: Exposure metrics with bioequivalence testing and dose proportionality
- *Population modeling*: Covariate selection, model comparison, parameter uncertainty quantification
- *Simulation-based design*: Dose optimization, sample size determination, protocol evaluation
- *Regulatory reporting*: Tables, listings, figures matching submission standards

== Learning Curve and Transition Path

=== Julia Language Adoption

Julia syntax is similar to MATLAB and Python:
- 1-based indexing
- Mathematical notation support (Greek letters, subscripts)
- Interactive REPL for exploration
- Dynamic typing with optional type annotations

Pharmacometricians with computational backgrounds (familiar with scripting in R, or MATLAB) typically become productive within 1-2 weeks.
Key concepts to internalize:
- Broadcasting with `.` operator for vectorization
- Multiple dispatch for function specialization
- Type stability for performance
- Package management with Project.toml/Manifest.toml

=== Resources for Learning

- *Julia Language Documentation*: docs.julialang.org @juliadocs for core language features and tutorials
- *Pumas Documentation*: docs.pumas.ai @pumas2024docs for pharmacometrics-specific guidance
- *Pumas Tutorials*: tutorials.pumas.ai @pumastutorials covers pharmacometric workflows from basic through advanced
- *AskPumas AI*: Interactive assistant for on-demand help with code examples and explanations

IDE support via VS Code with Julia extension provides professional development environment comparable to RStudio for R.

== Limitations and Considerations

Pumas adoption involves trade-offs:

*Learning Investment*: Requires learning Julia syntax and Pumas-specific conventions. Teams comfortable with established workflows (NONMEM, R)
face transition costs.

*Community Size*: Smaller than R or NONMEM communities, though growing. Fewer third-party packages and Stack Overflow answers than mature platforms.

*Organizational Inertia*: Pharmaceutical companies have established standard operating procedures (SOPs) around existing tools. Adopting new platform
requires validation, SOP updates, and training.

*Ecosystem Maturity*: Julia ecosystem is younger than R or Python. Some specialized tools may not yet have Julia equivalents.

These limitations are balanced by performance, reproducibility, and integration benefits. Organizations report reduced analysis time and improved
model development efficiency after transition period. The transition itself is supported by continuous improvements to learning resources, support through the discussion forum, and dedicated support offers.

== Future Directions and Outlook

Pharmacometric analyses grow in complexity—multi-endpoint models, machine learning integration, large-scale simulation, real-world evidence.
Unified platforms become necessary infrastructure as analyses span more data types, modeling approaches, and decision contexts.

Pumas demonstrates that comprehensive integration is achievable without sacrificing performance, flexibility, or rigorous statistical methods.
The platform enables quantitative drug development workflows from early discovery through regulatory submission within a single computational framework.

As model-informed drug development @marshall2016midd @barrett2022midd (MIDD) becomes standard practice, platforms integrating data science, mechanistic modeling, simulation, and
decision support will be essential tools for pharmaceutical development teams.

== Conclusion

This article introduced the Pumas platform through a complete pharmacometric workflow: multiple ascending dose study analyzed from raw data through
model-based dose selection. We demonstrated:
- Integrated data preparation and exploratory analysis (Section 3)
- Declarative model specification with `@model` macro (Section 4)
- Progressive estimation and comprehensive diagnostics (Section 5)
- Multi-endpoint PK-PD modeling with exposure-response analysis (Section 6)
- Simulation-based decision support for dose selection (Section 7)

The workflow executed entirely within Pumas, eliminating tool-switching overhead and data conversion steps that introduce errors and slow iteration.
Input validation caught errors at construction rather than during estimation. Consistent API patterns enabled rapid learning and transfer across analysis types.
High performance supported interactive model development and extensive simulation.

The complete annotated workflow appears in the Appendix with full code for data simulation, model definitions, diagnostic procedures, and simulation analyses.

Pumas provides production-ready infrastructure for pharmacometric analysis across drug development phases. Integration across data manipulation, statistical modeling,
simulation, and reporting enables rigorous quantitative decision-making from first-in-human studies through regulatory submission and post-marketing evaluation.

== Acknowledgements

This paper and, more importantly, the methodology and software development behind it was influenced by many. We thank Joga Gobburu for his initial leadership and investment that made this possible. We thank Simon Byrne, José Bayoán Santiago Calderón, Yingbo Ma, Vaibhav Dixit, Chris Elrod, Xingjian Guo, Shubham Maddhashiya, Joakim Nyberg for contributions in the form of both direct software development and discussions that ultimately materialized here. We thank the rest of the PumasAI development team for building the software foundation on which this application stands. We would also like to thank the early adopters and beta testers of Pumas who provided invaluable feedback that shaped the platform. Finally, we thank the broader pharmacometrics community for their ongoing contributions to the field, which continue to inspire innovation. Last but not the least, the invaluable feedback provided by the members of the PumasAI consulting team during the early days of the Pumas platform development helped shape the direction of this work.

Claude Sonnet 4.5, via GitHub Copilot was utilized in the preparation of this document in several ways, including a critical review of the narrative, text, and methodology, as well as language editing for clarity, correctness, and conciseness.

== Author Contributions

*Vijay Ivaturi:* Conceptualization, Writing - Original Draft, Writing - Review & Editing

*Patrick Mogenson:* Conceptualization, Writing - Original Draft, Writing - Review & Editing

*Andreas Noack*: Reviewing & Editing

*Mohamed Tarek:* Reviewing & Editing

*David Müller-Widmann*: Reviewing & Editing

*Julius Krumbiegel*: Reviewing & Editing

*Michael Hatherly*: Reviewing & Editing

*Niklas Korsbo*: Reviewing & Editing

== Conflicts of Interest

All authors are currently or recently employed by PumasAI, which sells software that leverages the methodology described herein.

== Funding

This work was funded by PumasAI

== Data and Code Availability

[TO BE COMPLETED: Repository links and availability statement. Journal requires sufficient technical details including example dataset and model code as supplementary material.]

