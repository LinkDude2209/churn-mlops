terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.35"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "churn-mlops"
  }
}

resource "kubernetes_deployment" "churn_api" {
  metadata {
    name      = "churn-api"
    namespace = kubernetes_namespace.mlops.metadata[0].name

    labels = {
      app = "churn-api"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "churn-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "churn-api"
        }
      }

      spec {
        container {
          name              = "churn-api"
          image             = "churn-api:v1"
          image_pull_policy = "Never"

          port {
            container_port = 8000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "churn_api" {
  metadata {
    name      = "churn-api-service"
    namespace = kubernetes_namespace.mlops.metadata[0].name
  }

  spec {
    selector = {
      app = "churn-api"
    }

    port {
      port        = 8000
      target_port = 8000
      node_port   = 30081
    }

    type = "NodePort"
  }
}