job "breastmarc-export-92-medyolo" {

  datacenters = ["cluster"]
  type = "batch"

  priority = 50

  meta {
    owner = "kgeissler"
  }

  constraint {
    attribute = "${node.unique.name}"
    operator = "regexp"
    value = "gpu-cluster-(03|04|05|06|07|08|09|10|11)"
  }

  affinity {
    attribute = "${node.unique.name}"
    operator = "regexp"
    value = "gpu-cluster-(09|10|11)"
    weight    = 100
  }

  group "export" {
    task "export" {
      leader = true
      driver = "docker"

      config {
        image = "registry.fme.lan/breastmarc/medyolo:latest"
        image_pull_timeout = "20m"
        shm_size = 16000000000
        command = "bash"

        args = [
          "-c",
          <<EOL
          python onnx_export.py \
          --weights runs/train/exp/weights/best.pt \
          --output runs/train/exp/weights/best.onnx \
          --img-size-x 256 \
          --img-size-y 256 \
          --img-size-z 256 \
          --input-channels 2
          EOL
        ]
        
        mount {
          type = "volume"
          target = "/MedYOLO/runs"
          source = "medyolo_runs_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/deep_learning/output/09_mass_nme_segmentation/92_medyolo/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
              }
            }
          }
        }

        mount {
          type = "volume"
          target = "/data"
          source = "data_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/deep_learning/input/medyolo_data/92_medyolo/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
              }
            }
          }
        }

        mount {
          type = "volume"
          target = "/config"
          source = "config_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/deep_learning/input/configs/09_mass_nme_segmentation/92_medyolo/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
              }
            }
          }
        }

        volumes = []
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
        memory = 20000
        memory_max = 30000
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
