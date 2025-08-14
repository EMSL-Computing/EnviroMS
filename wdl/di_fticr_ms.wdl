version 1.0

workflow fticrmsNOM {
    input {
        String? docker_image  # Optional input for Docker image
    }

    call runDirectInfusion {
        input:
            docker_image = docker_image
    }

    output {
        String out = runDirectInfusion.out
        Array[File] output_files = runDirectInfusion.output_files
        Array[File] van_krevelen_plots = runDirectInfusion.van_krevelen_plots
        Array[File] dbe_vs_c_plots = runDirectInfusion.dbe_vs_c_plots
        Array[File] ms_class_plots = runDirectInfusion.ms_class_plots
        Array[File] mz_error_class_plots = runDirectInfusion.mz_error_class_plots
        Array[File] qc_plots = runDirectInfusion.qc_plots
    }
}

task runDirectInfusion {
    input {
        Array[File] file_paths
        String output_directory
        String output_type
        File corems_toml_path
        String polarity
        Int raw_file_start_scan
        Int raw_file_final_scan
        Boolean is_centroid
        File calibration_ref_file_path
        Boolean calibrate
        Boolean plot_mz_error
        Boolean plot_ms_assigned_unassigned
        Boolean plot_c_dbe
        Boolean plot_van_krevelen
        Boolean plot_ms_classes
        Boolean plot_mz_error_classes
        Int jobs_count = 1
        String? docker_image
    }

    command {
        enviroMS run_di_wdl \
            ${sep=',' file_paths} \
            ${output_directory} \
            ${output_type} \
            ${corems_toml_path} \
            ${polarity} \
            ${raw_file_start_scan} \
            ${raw_file_final_scan} \
            ${is_centroid} \
            ${calibration_ref_file_path} \
            -c ${calibrate} \
            -e ${plot_mz_error} \
            -a ${plot_ms_assigned_unassigned} \
            -cb ${plot_c_dbe} \
            -vk ${plot_van_krevelen} \
            -mc ${plot_ms_classes} \
            -ec ${plot_mz_error_classes} \
            --jobs ${jobs_count}
    }

    output {
        String out = read_string(stdout())
        Array[File] output_files = glob('${output_directory}/**/*.*')
        Array[File] van_krevelen_plots = glob('${output_directory}/**/van_krevelen/*.*')
        Array[File] dbe_vs_c_plots = glob('${output_directory}/**/dbe_vs_c/*.*')
        Array[File] ms_class_plots = glob('${output_directory}/**/ms_class/*.*')
        Array[File] mz_error_class_plots = glob('${output_directory}/**/mz_error_class/*.*')
        Array[File] qc_plots = glob('${output_directory}/**/qc_plots/*.*')
    }

    runtime {
        docker: "~{if defined(docker_image) then docker_image else 'microbiomedata/enviroms:5.0.0'}"
    }
}
