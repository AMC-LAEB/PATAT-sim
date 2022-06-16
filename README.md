
# Propelling Action for Testing and Treating (PATAT) model
---

PATAT is a stochastic agent-based modelling framework developed to investigate the impact of using SARS-CoV-2 antigen rapid diagnostic tests (Ag-RDTs) in communities with demographic profiles, contact mixing patterns, levels of public health resources mirroring those often observed in low- and middle-income countries (LMICs). PATAT has been used in the Phase 2 study of the Access to COVID-19 Tools Accelerator (ACT-Accelerator) SARS-CoV-2 diagnostics modelling consortium, led by the Foundation for Innovative New Diagnostics (FIND). This study interrogates how different Ag-RDT availability and distribution strategies, including the implementation of community testing in households, schools, formal workplaces and regular mass gatherings (e.g. churches), as well as post-testing behavioural changes and public health interventions could impact onward disease transmissions and pandemic mitigation in LMICs. This original scientific paper describing this work is under review:
> Strategies for using antigen rapid diagnostic tests to reduce transmission of SARS-CoV-2 in low- and middle-income countries: a mathematical modelling study. Alvin X. Han, Sarah Girdwood, Shaukat Khan, Jilian A. Sacks, Amy Toporowski, MD, Naushin Huq, Emma Hannay, Colin A. Russell, Brooke E. Nichols. (2022).

The analyses codes for this study can be found as a Jupyter Notebook [here](https://github.com/AMC-LAEB/PATAT-sim/blob/main/projects/ACTA_phase2/han-et-al_ACTA_phase2.ipynb). As the raw simulation data are very large in size (>200Gb), we have provided processed data files necessary to run the Jupyter Notebook.

---

PATAT can also simulate the spread of up to two SARS-CoV-2 variants with different viral and transmission properties (e.g. incubation and infection periods, maximum within-host viral load, disease severity, relative transmissibility, etc.). As such, PATAT can also be used to perform other COVID-19 analyses.

We have used PATAT to investigate how testing capacities, sampling coverage and sequencing proportions jointly impact the effectiveness of pathogen surveillance. This work and the original scientific paper describing PATAT can be found [here](https://doi.org/10.1101/2022.05.20.22275319), currently as a pre-print and cited as:
> Low testing rates limit the ability of genomic surveillance programs to monitor SARS-CoV-2 variants: a mathematical modelling study. Alvin X. Han, Amy Toporowski, Jilian A. Sacks, Mark D. Perkins, Sylvie Briand, Maria van Kerkhove, Emma Hannay, Sergio Carmona, Bill Rodriguez, Edyth Parker, Brooke E. Nichols, Colin A. Russell. (2022). medRxiv,  2022.05.20.22275319.

The analyses codes for this paper can be found as a Jupyter Notebook [here](https://github.com/AMC-LAEB/PATAT-sim/blob/main/projects/surveillance/han-et-al_genome_surveillance_lmics.ipynb). Similarly, as the raw simulation data are extremely large in size (>400Gb), we have provided processed data files/folders (which still total to >8Gb) that can be download them separately [here](https://www.dropbox.com/sh/gcxdh6yvzgmng4h/AAAoy0vnlJy0Tg-VoQJ4jzRHa?dl=0). Have all processed data files/folders in the same directory as the Jupyter Notebook before running it.  

If you have any queries about the model and analyses, please contact <x.han@amsterdamumc.nl>.

---

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
