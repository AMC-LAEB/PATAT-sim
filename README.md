
# Propelling Action for Testing and Treating (PATAT) model
---
PATAT is a stochastic agent-based modelling framework developed to investigate the impact of using SARS-CoV-2 antigen rapid diagnostic tests (Ag-RDTs) in communities with demographic profiles, contact mixing patterns, levels of public health resources mirroring those often observed in low- and middle-income countries (LMICs). PATAT has been used in the Phase 2 study of the Access to COVID-19 Tools Accelerator (ACT-Accelerator) SARS-CoV-2 diagnostics modelling consortium, led by the Foundation for Innovative New Diagnostics (FIND). This study interrogates how different Ag-RDT availability and distribution strategies, including the implementation of community testing in households, schools, formal workplaces and regular mass gatherings (e.g. churches), as well as post-testing behavioural changes and public health interventions could impact onward disease transmissions and pandemic mitigation

The analyses codes for the ACT-Accelerator Phase 2 study be found as a Jupyter Notebook [here](https://github.com/AMC-LAEB/PATAT-sim/tree/main/ACTA_phase2)

PATAT can also simulate the spread of up to two SARS-CoV-2 variants with different viral and transmission properties (e.g. incubation and infection periods, maximum within-host viral load, disease severity, relative transmissibility, etc.). As such, PATAT can also be used to perform other COVID-19 analyses.

We have used PATAT to investigate how testing capacities, sampling coverage and sequencing proportions jointly impact the effectiveness of pathogen surveillance. This work and the original scientific paper describing PATAT can be found [here](https://link) and cited as:
>

The analyses codes for this paper can be found as a Jupyter Notebook [here](https://github.com/AMC-LAEB/PATAT-sim/tree/main/surveillance)

If you have questions or comments, please email to <x.han@amsterdamumc.nl>.

## Requirements and installation of PATAT-sim
---
```PATAT-sim``` minimally requires Python 3.7 and Cython 0.29.23. Before installation, install Cython by ```pip```:

```pip install cython```

```PATAT-sim``` can also be installed as a ```pip``` package:  
```pip install PATAT-sim```

If installation by ```pip``` fails, you can also clone this repository in your local drive and install by:

```
git clone https://github.com/AMC-LAEB/PATAT-sim
cd PATAT-sim
python setup.py install --record installed_files.txt
```

## Basic usage
---
### Simulate SARS-CoV-2 epidemic
1. Fill in demographic and transmission parameters in the example spreadsheet provided (```patat_input_file.xlsx```).
2. Run simulation for _N_ number of days:
```runpatat.py simulate --input patat_input_file.xlsx --ndays N```

### Simulate genomic surveillance from previously simulated epidemics
1. Suppose you had previously used PATAT to simulate an epidemic and the results are stored in the path ```PATAT-sim_output```.
2. Run genomic surveillance simulations (```N``` number of random boostraps) assuming ```f``` fraction of healthcare facilities as tertiary facilities (i.e. assumed that only tertiary facilties have capacities and resources to perform genome sequencing), ```s``` fraction of samples collected are being sequenced weekly:
```runpatat.py gs --resfolder PATAT-sim_output --tertiary_hcf_prop f --seq_prop s --gs_sim_N N```
