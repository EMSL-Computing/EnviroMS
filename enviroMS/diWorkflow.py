import cProfile
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import click
import toml
from corems.encapsulation.factory.processingSetting import (
    MolecularFormulaSearchSettings,
)
from corems.encapsulation.input.parameter_from_json import (
    load_and_set_toml_parameters_class,
)
from corems.mass_spectrum.calc.Calibration import MzDomainCalibration
from corems.mass_spectrum.input.massList import ReadMassList
from corems.molecular_id.factory.classification import HeteroatomsClassification
from corems.molecular_id.factory.MolecularLookupTable import MolecularCombinations
from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
from corems.molecular_id.search.priorityAssignment import OxygenPriorityAssignment
from corems.transient.input.brukerSolarix import ReadBrukerSolarix
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from tqdm import tqdm
import pandas as pd
import seaborn as sns


@dataclass
class DiWorkflowParameters:
    # input type: masslist, bruker_transient, thermo_reduced_profile

    # scans to sum for thermo raw data, reduce profile
    raw_file_start_scan: int = 1
    raw_file_final_scan: int = 7

    # input output paths
    file_paths: tuple = ("data/...", "data/...")
    output_directory: str = "data/..."
    output_group_name: str = "..."
    output_type: str = "csv"

    # polarity for masslist input
    polarity: int = -1
    is_centroid: bool = False
    # corems settings
    corems_toml_path: str = "configuration/di_corems.toml"
    # calibration
    calibrate: bool = True
    calibration_ref_file_path: str = "data/raw_data/SRFA.ref"

    # plots
    plot_mz_error: bool = True
    plot_ms_assigned_unassigned: bool = True

    plot_c_dbe: bool = True
    plot_van_krevelen: bool = True
    plot_ms_classes: bool = True
    plot_mz_error_classes: bool = True
    plot_qc: bool = True

    def to_toml(self):
        return toml.dumps(asdict(self))


def run_thermo_reduce_profile(file_location, corems_params_path, first_scan, last_scan):
    print("run thermo reduce profile")
    from corems.mass_spectra.input import rawFileReader

    parser = rawFileReader.ImportMassSpectraThermoMSFileReader(file_location)
    parser.chromatogram_settings.scans = (first_scan, last_scan)

    mass_spectrum = parser.get_average_mass_spectrum()
    return mass_spectrum


def run_bruker_transient(file_location, corems_params_path):
    print("run bruker transient")
    with ReadBrukerSolarix(file_location) as transient:
        transient.set_parameter_from_toml(corems_params_path)
        mass_spectrum = transient.get_mass_spectrum(
            plot_result=False, auto_process=True
        )

    return mass_spectrum


def read_fticr_raw_data(file_location, workflow_params):
    file_path = Path(file_location)

    if file_path.suffix == ".raw":
        mass_spectrum = run_thermo_reduce_profile(
            file_location, workflow_params, 
            first_scan = workflow_params.raw_file_start_scan, 
            last_scan = workflow_params.raw_file_final_scan
        )

    elif file_path.suffix == ".d":
        mass_spectrum = run_bruker_transient(
            file_location, workflow_params.corems_toml_path
        )

    elif (
        file_path.suffix == ".txt"
        or file_path.suffix == "csv"
        or file_path.suffix == ".tsv"
    ):
        mass_spectrum = get_masslist(
            file_location,
            workflow_params.corems_toml_path,
            polarity=workflow_params.polarity,
            is_centroid=workflow_params.is_centroid,
        )
    
    return mass_spectrum


def get_masslist(file_location, corems_params_path, polarity, is_centroid):
    print("get masslist")
    if is_centroid:
        reader = ReadMassList(file_location)
    else:
        reader = ReadMassList(
            file_location, header_lines=7, isCentroid=False, isThermoProfile=True
        )

    reader.set_parameter_from_toml(parameters_path=corems_params_path)

    return reader.get_mass_spectrum(polarity=polarity)


def run_assignment(file_location, workflow_params):
    print("run assignment")

    mass_spectrum = read_fticr_raw_data(file_location, workflow_params)

    mass_spectrum.set_parameter_from_toml(workflow_params.corems_toml_path)

    if workflow_params.calibrate:
        ref_file_location = Path(workflow_params.calibration_ref_file_path)

        MzDomainCalibration(mass_spectrum, ref_file_location).run()

    # force it to one job. daemon child can not have child process
    mass_spectrum.molecular_search_settings.db_jobs = 1

    SearchMolecularFormulas(mass_spectrum, first_hit=False).run_worker_mass_spectrum()

    return mass_spectrum


def generate_database(corems_parameters_file, jobs):
    print("generate database")
    """Create molecular formula database.
    corems_parameters_file: Path for CoreMS TOML Parameters file
    --jobs: Number of processes to run
    """
    click.echo("Loading Searching Settings from %s" % corems_parameters_file)

    molecular_search_settings = load_and_set_toml_parameters_class(
        "MolecularSearch",
        MolecularFormulaSearchSettings(),
        parameters_path=corems_parameters_file,
    )
    molecular_search_settings.db_jobs = jobs
    molecular_search_settings.url_database = None
    MolecularCombinations().runworker(molecular_search_settings)


def read_workflow_parameter(di_workflow_parameters_toml_file):
    print("read workflow parameter")
    with open(di_workflow_parameters_toml_file, "r") as infile:
        return DiWorkflowParameters(**toml.load(infile))


def create_plots(mass_spectrum, workflow_params, dirloc):
    print("create plots")
    ms_by_classes = HeteroatomsClassification(
        mass_spectrum, choose_molecular_formula=False
    )

    if workflow_params.plot_ms_assigned_unassigned:
        print("Plotting assigned vs. unassigned mass spectrum")
        ax_ms = ms_by_classes.plot_ms_assigned_unassigned()
        plt.savefig(dirloc / "assigned_unassigned.png", bbox_inches="tight")
        plt.clf()

    if workflow_params.plot_mz_error:
        print("Plotting mz_error")
        ax_ms = ms_by_classes.plot_mz_error()
        plt.savefig(dirloc / "mz_error.png", bbox_inches="tight")
        plt.clf()

    if workflow_params.plot_van_krevelen:
        van_krevelen_dirloc = dirloc / "van_krevelen"
        van_krevelen_dirloc.mkdir(exist_ok=True, parents=True)

    if workflow_params.plot_c_dbe:
        c_dbe_dirloc = dirloc / "dbe_vs_c"
        c_dbe_dirloc.mkdir(exist_ok=True, parents=True)

    if workflow_params.plot_ms_classes:
        ms_class_dirloc = dirloc / "ms_class"
        ms_class_dirloc.mkdir(exist_ok=True, parents=True)

    if workflow_params.plot_mz_error_classes:
        mz_error_class_dirloc = dirloc / "mz_error_class"
        mz_error_class_dirloc.mkdir(exist_ok=True, parents=True)

    if workflow_params.plot_qc:
        qc_plot_dirloc = dirloc / "qc_plots"
        qc_plot_dirloc.mkdir(exist_ok=True, parents=True)

    pbar = tqdm(ms_by_classes.get_classes())

    for classe in pbar:
        pbar.set_description_str(
            desc="Plotting results for class {}".format(classe), refresh=True
        )

        if workflow_params.plot_van_krevelen:
            ax_c = ms_by_classes.plot_van_krevelen(classe)
            plt.savefig(
                van_krevelen_dirloc / "{}.png".format(classe), bbox_inches="tight"
            )
            plt.clf()

        if workflow_params.plot_mz_error_classes:
            ax_c = ms_by_classes.plot_mz_error_class(classe)
            plt.savefig(
                mz_error_class_dirloc / "{}.png".format(classe), bbox_inches="tight"
            )
            plt.clf()

        if workflow_params.plot_ms_classes:
            ax_c = ms_by_classes.plot_ms_class(classe)
            plt.savefig(ms_class_dirloc / "{}.png".format(classe), bbox_inches="tight")
            plt.clf()

        if workflow_params.plot_c_dbe:
            ax_c = ms_by_classes.plot_dbe_vs_carbon_number(classe)
            plt.savefig(c_dbe_dirloc / "{}.png".format(classe), bbox_inches="tight")
            plt.clf()
    
    # Create QC plots
    if workflow_params.plot_qc:
        ms_df = mass_spectrum.to_dataframe()
        qc_fig, qc_axes = create_qc_figure(mass_spectrum, ms_df, title=mass_spectrum.sample_name,  hspace=0.25, wspace=0.35)
        qc_fig.savefig(qc_plot_dirloc / "{}_qc.png".format(mass_spectrum.sample_name), dpi=100, bbox_inches='tight')
        plt.close(qc_fig)
        plt.close('all')


def create_qc_figure(msobj, msdf, title='QC Plot', figsize=(24, 10), nrows=2, ncols=4, hspace=0.22, wspace=0.22):
    """
    Creates a QC matplotlib figure with a specified gridspec layout.
    Thread-safe: returns a new figure and axes on each call.
    
    Parameters:
        figsize (tuple): Size of the full figure (width, height).
        nrows (int): Number of subplot rows.
        ncols (int): Number of subplot columns.
        hspace (float): Height spacing between rows.
        wspace (float): Width spacing between columns.
        
    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        axes (list of matplotlib.axes._subplots.AxesSubplot): List of axes.
    """
    
    # ensure that key element columns were initiated or it could crash:
    key_elements = ['C','H','O','N','S','P']
    for ele in key_elements:
        if ele not in msdf.columns:
            msdf[ele] = 0
    msdf[key_elements] = msdf[key_elements].fillna(0)
    
    mz = msobj.mz_cal_profile
    abu = msobj.abundance_profile
    
    def subset_mz(mz_array, abu_array, mz_min, mz_max):
        idx = (mz_array >= mz_min) & (mz_array <= mz_max)
        return mz_array[idx], abu_array[idx]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=hspace, wspace=wspace)
    axes = [fig.add_subplot(gs[row, col]) for row in range(nrows) for col in range(ncols)]
    
    
    ###########
    # Mass Spectrum
    ############
    # Full spectrum plot
    axes[0].plot(mz, abu, lw=1,c='k')
    
    # Zoom 1
    mz_zoom1, abu_zoom1 = subset_mz(mz, abu, 200, 800)
    axes[1].plot(mz_zoom1, abu_zoom1, lw=1, c='k')  # y-limits autoscale
    axes[1].set_ylim(0, 0.1*max(abu))
    axes[1].set_xlim(200, 800)
    
    # Zoom 2
    mz_zoom2, abu_zoom2 = subset_mz(mz, abu, 282.95, 283.2)
    axes[4].plot(mz_zoom2, abu_zoom2, lw=1, c='k')
    
    # Zoom 4
    mz_zoom3, abu_zoom3 = subset_mz(mz, abu, 571.0,571.25)
    axes[5].plot(mz_zoom3, abu_zoom3, lw=1, c='k')

    intensity_label = 'Intensity (a.u.)'
    axes[0].set_ylabel(intensity_label)
    axes[1].set_ylabel(intensity_label)
    axes[4].set_ylabel(intensity_label)
    axes[5].set_ylabel(intensity_label)

    mass_label = '$m/z$'
    axes[0].set_xlabel(mass_label)
    axes[1].set_xlabel(mass_label)
    axes[4].set_xlabel(mass_label)
    axes[5].set_xlabel(mass_label)
    

    #########
    # Error Plot
    #########
    
    axes[2].scatter(msdf['m/z'], msdf['m/z Error (ppm)'], s=msdf['S/N']*0.01, c='k',alpha=0.5)
    axes[2].set_ylabel('$m/z$ Error (ppm)')
    axes[2].set_xlabel(mass_label)

    #########
    # Van Krevelen 
    #########
    
    axes[3].scatter(msdf['O/C'], msdf['H/C'], s=msdf['S/N']*0.01, c='k',alpha=0.5)
    axes[3].set_ylabel('H/C')
    axes[3].set_xlabel('O/C')
    axes[3].set_xlim(0, 1.25)
    axes[3].set_ylim(0.25,2.25)
    
    #########
    # Heteroatom Countplot
    #########
    # Filter the dataframe
    df_filtered = msdf[
        (msdf['Heteroatom Class'] != 'unassigned') &
        (msdf['Is Isotopologue'] == 0)
    ].copy()
    
    # Classify entries based on heteroatom presence
    def group_hetero(row):
       if row['S'] > 0:
           return 'S > 0'
       elif row['N'] > 0:
           return 'N > 0'
       #elif row['P'] > 0: # No P in MAOM
       #    return 'P > 0'
       else:
           return 'No S/N'#'/P'
       
    df_filtered['Hetero Group'] = df_filtered.apply(group_hetero, axis=1)
    # Sort O values numerically for better x-axis order
    df_filtered['O'] = pd.to_numeric(df_filtered['O'], errors='coerce').fillna(0).astype(int)
    
    # Sort hetero groups in desired priority
    #group_order = ['No S/N/P', 'S > 0', 'N > 0', 'P > 0']
    # if excluding P from annotation
    group_order = ['No S/N', 'S > 0', 'N > 0']

    sns.countplot(
        data=df_filtered,
        x='O',
        hue='Hetero Group',
        order=sorted(df_filtered['O'].unique()),
        hue_order=group_order,
        ax=axes[6]
    )
    
    axes[6].tick_params(axis='x', labelsize=10, rotation=45)
    
    legend = axes[6].legend(
            title='Hetero Group',
            fontsize=8,
            title_fontsize=9,
            loc='best',
            frameon=True,
            borderpad=0.3,
            handletextpad=0.3,
            borderaxespad=0.3,
            labelspacing=0.3,
            handlelength=1
        )

    #########
    # NOSC KDE
    #########

    def NOSCcalc(df):
        NOSC = 4 - ((4*df['C'] + df['H'] - 2*df['O'] - 3*df['N'] + 5*df['P'] - 2*df['S']) / df['C'])
        return NOSC

    msdf['NOSC'] = NOSCcalc(msdf)
    sns.kdeplot(data=msdf, x='NOSC',ax=axes[7], c= 'k')
    axes[7].set_xlim(-2.5,2.5)
    
    fig.suptitle(title, fontsize=24, y=0.95)
    
    print("made qc plots")
    return fig, axes


def workflow_worker(args):
    print("workflow worker")
    file_location, workflow_params_toml_str = args

    workflow_params = DiWorkflowParameters(**toml.loads(workflow_params_toml_str))

    mass_spec = run_assignment(file_location, workflow_params)

    dirloc = Path(workflow_params.output_directory) / mass_spec.sample_name

    dirloc.mkdir(exist_ok=True, parents=True)

    output_path = dirloc / mass_spec.sample_name

    eval(
        "mass_spec.to_{OUT_TYPE}(output_path)".format(
            OUT_TYPE=workflow_params.output_type
        )
    )

    create_plots(mass_spec, workflow_params, dirloc)

    return "Success" + str(os.getpid())


# Resource profiling. not used.
def cprofile_worker(file_location, workflow_params_toml_str):
    cProfile.runctx(
        "run_assignment(file_location, workflow_params)",
        globals(),
        locals(),
        "di-fticr-di.prof",
    )
    # stats = pstats.Stats("topics.prof")
    # stats.strip_dirs().sort_stats("time").print_stats()


def run_wdl_direct_infusion_workflow(*args, **kwargs):
    print("run wdl direct infusion workflow")
    cores = kwargs.get("jobs")
    del kwargs["jobs"]
    kwargs["polarity"] = -1 if kwargs.get("polarity") == "negative" else 1

    workflow_params = DiWorkflowParameters(**kwargs)
    workflow_params.file_paths = workflow_params.file_paths.split(",")
    # workflow_params.output_directory = kwargs.get['output_directory']
    # workflow_params.output_type = kwargs.get['output_type']
    # workflow_params.corems_toml_path = kwargs.get['corems_toml_path']
    # workflow_params.polarity = -1 if kwargs.get['polarity'] == 'negative' else 1
    # workflow_params.raw_file_start_scan = kwargs.get['raw_file_start_scan']
    # workflow_params.raw_file_final_scan = kwargs.get['raw_file_final_scan']
    # workflow_params.is_centroid = kwargs.get['is_centroid']
    # workflow_params.calibration_ref_file_path = kwargs.get['calibration_ref_file_path']

    # workflow_params.calibrate = kwargs.get['calibrate']
    # workflow_params.calibrate = kwargs.get['plot_mz_error']
    # workflow_params.calibrate = kwargs.get['plot_ms_assigned_unassigned']
    # workflow_params.calibrate = kwargs.get['plot_c_dbe']
    # workflow_params.calibrate = kwargs.get['plot_van_krevelen']
    # workflow_params.calibrate = kwargs.get['plot_ms_classes']
    # workflow_params.calibrate = kwargs.get['plot_mz_error_classes']
    # workflow_params.calibrate = kwargs.get['calibrate']

    dirloc = Path(workflow_params.output_directory)
    dirloc.mkdir(exist_ok=True)

    pool = Pool(cores)

    worker_args = [
        (file_path, workflow_params.to_toml())
        for file_path in workflow_params.file_paths
    ]
    file_path = Path(worker_args[0][0])
    print(file_path)
    for worker_arg in worker_args:
       workflow_worker(worker_arg)

    # multithreading doesn't work for some reason
    # for i, results in enumerate(pool.imap_unordered(workflow_worker, worker_args), 1):
    #     pass
    print("hellooooo")
    pool.close()
    pool.join()


def run_direct_infusion_workflow(workflow_params_file, jobs, replicas):
    print("run direct infusion workflow")

    click.echo("Loading Searching Settings from %s" % workflow_params_file)
    workflow_params = read_workflow_parameter(workflow_params_file)

    # Set up paths
    dirloc = Path(workflow_params.output_directory)
    dirloc.mkdir(exist_ok=True)

    worker_args = replicas * [
        (file_path, workflow_params.to_toml())
        for file_path in workflow_params.file_paths
    ]

    cores = jobs
    pool = Pool(cores)

    for worker_arg in worker_args:
        workflow_worker(worker_arg)
    # for i, results in enumerate(pool.imap_unordered(workflow_worker, worker_args), 1):

    #    pass

    pool.close()
    pool.join()


def run_di_mpi(workflow_params_file, tasks, replicas):
    import os
    import sys

    from mpi4py import MPI

    # from mpi4py.futures import MPIPoolExecutor
    sys.path.append(os.getcwd())

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    workflow_params = read_workflow_parameter(workflow_params_file)
    all_worker_args = replicas * [
        (file_path, workflow_params.to_toml())
        for file_path in workflow_params.file_paths
    ]

    # worker_args = comm.scatter(all_worker_args, root=0)

    # will only run tasks up to the number of files paths selected in the EnviroMS File
    if len(all_worker_args) <= size:
        workflow_worker(all_worker_args[0])

    else:
        print(
            "Tasks needs to be the same size of the input data count, until you find time to come and help to code this section :D"
        )
