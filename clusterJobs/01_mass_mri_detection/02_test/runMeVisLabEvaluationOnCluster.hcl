job "breastmarc-MeVisLab_muellergroh" {

  datacenters = ["cluster"]
  type = "batch"
  priority = 60

  meta {
    owner = "muellergroh"
  }

  affinity {
    attribute = "${node.unique.name}"
    operator = "regexp"
    value = "cluster-node-(gamma|delta)"
    weight    = 100
  }

  group "task-group" {
    network {
      port "vnc" {to = 5900}
    }

    volume "mlab_license" {
      type      = "host"
      read_only = true
      source    = "dl_mlab_license"
    }

    volume "ssl_certificates" {
      type      = "host"
      read_only = true
      source    = "ssl_certificates"
    }

    task "MeVisLab" {

      driver = "docker"

      volume_mount {
        volume = "mlab_license"
        destination = "/MeVisLab/License.dat"
        read_only = true
      }

      volume_mount {
        volume = "ssl_certificates"
        destination = "/etc/ssl/certs"
        read_only = true
      }

      config {
        image = "registry.fme.lan/breastmarc/mlab-breastmarc-sdk:latest"

        image_pull_timeout = "20m"

        args = [
          "/Projects/MasterThesisYOLO3DEvaluation/evaluate_laurens_yolo3d.mlab"
        ]

        ports = ["vnc"]

        # mount {
        #   type = "volume"
        #   target = "/deep_learning"
        #   source = "deep_learning_${NOMAD_ALLOC_ID}"
        #   readonly = false
        #   volume_options {
        #     driver_config {
        #       name = "local"
        #       options {
        #         type = "cifs"
        #         device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/deep_learning/"
        #         o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
        #       }
        #     }
        #   }
        # }

        mount {
          type = "volume"
          target = "/Projects"
          source = "projects_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/Projects/"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
              }
            }
          }
        }

        mount {
          type = "volume"
          target = "/workspace"
          source = "workspace_${NOMAD_ALLOC_ID}"
          readonly = false
          volume_options {
            driver_config {
              name = "local"
              options {
                type = "cifs"
                device = "//gaia.intern.mevis.fraunhofer.de/externalServices/satori-breastmarc/production_app/workspace"
                o = "file_mode=0775,dir_mode=0777,noperm,nopersistenthandles,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
              }
            }
          }
        }

        # mount {
        #   type = "volume"
        #   target = "/duke"
        #   source = "duke_${NOMAD_ALLOC_ID}"
        #   readonly = false
        #   volume_options {
        #     driver_config {
        #       name = "local"
        #       options {
        #         type = "cifs"
        #         device = "//gaia.intern.mevis.fraunhofer.de/industry/Siemens/BreastMARC/GJ2021/data/Duke/"
        #         o = "file_mode=0775,dir_mode=0777,noperm,addr=gaia.intern.mevis.fraunhofer.de,username=${MOUNT_USERNAME},password=${MOUNT_PASSWORD},domain=fme"
        #       }
        #     }
        #   }
        # }

        volumes = [
        #   "/ssd/docker/volumes/deep_learning_${NOMAD_ALLOC_ID}/_data/input:/input",
        #   "/ssd/docker/volumes/deep_learning_${NOMAD_ALLOC_ID}/_data/output:/output",
          "/ssd/docker/volumes/deep_learning_${NOMAD_ALLOC_ID}/_data/model_repository:/model_repository:ro"
        ]
      }

      env {
        MLAB_FMEProjects_BreastMARC_ModelRepo = "/model_repository"
        START_VNC = "true"
        SSL_CERT_FILE = "/etc/ssl/certs/ca-certificates.crt"
        REQUESTS_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
        ML_CACHE_SIZE_IN_KB = 20000000
        OUTPUT_CSV_PATH = "/Projects/MasterThesisYOLO3DEvaluation/results/02_test-e500.csv"
        CONFIG_FILE_PATH = "/Projects/MasterThesisYOLO3DEvaluation/configs/02_test_BreastMARC_2025_05_09.json"
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
        cpu = 5000
        memory = 30000
        memory_max = 150000

        #device "nvidia/gpu" {
        #  count = 1
        #}
      }
    }
  }
}
