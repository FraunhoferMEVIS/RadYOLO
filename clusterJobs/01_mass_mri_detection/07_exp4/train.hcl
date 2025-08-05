job "breastmarc-train-masterthesis-yolo11-exp4" {

  datacenters = ["cluster"]
  type = "batch"

  priority = 50

  meta {
    owner = "muellergroh"
  }

  constraint {
    attribute = "${node.unique.name}"
    operator  = "regexp"
    value     = "gpu-cluster-(03|04|05|06|07|08|09|10|11)"
  }

  affinity {
    attribute = "${node.unique.name}"
    operator  = "regexp"
    value     = "gpu-cluster-(09|10|11)"
    weight    = 100
  }

  group "train" {
    task "train" {
      leader = true
      driver = "docker"

      config {
        image               = "registry.fme.lan/breastmarc/masterthesis-yolo3d:20250702-rev1"
        image_pull_timeout  = "20m"
        shm_size            = 24000000000
        
        mount {
          type = "volume"
          target = "/app/runs"
          source = "yolo11_3d_runs_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/MasterThesisYOLO3D/output/01_mass_mri_detection/07_exp4/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme-intern"
              }
            }
          }
        }

        mount {
          type = "volume"
          target = "/app/data"
          source = "data_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/MasterThesisYOLO3D/input/datasets/01_mass_mri_detection/01_test/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme-intern"
              }
            }
          }
        }

        mount {
          type = "volume"
          target = "/app/config"
          source = "config_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/MasterThesisYOLO3D/input/configs/01_mass_mri_detection/07_exp4/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme-intern"
              }
            }
          }
        }

        volumes = []
      }

      env {
        YOLO113D_TRAIN_CONFIG = "/app/config/train.yaml"
      }

      template {
        data = <<EOH
MOUNT_USERNAME = "{{with secret "secret/data/nonusers/breastmarc-intern-nu"}}{{.Data.data.user}}{{end}}"
MOUNT_PASSWORD = "{{with secret "secret/data/nonusers/breastmarc-intern-nu"}}{{.Data.data.password}}{{end}}"
        EOH

        destination = "secrets/file.env"
        env         = true
        change_mode = "noop"
      }
      vault {
        policies = ["nonuser-secrets-breastmarc-intern-nu"]
      }

      resources {
        cpu = 10000
        memory = 85000
        memory_max = 100000
        device "nvidia/gpu" {
          count = 1
        }
      }
    }

    restart {
      attempts = 0
      mode = "fail"
    }
  }

  reschedule {
    attempts = 0
  }
}
