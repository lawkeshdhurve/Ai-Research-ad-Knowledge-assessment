from typing import Tuple, List
import numpy as np

# Domain Categories
CATEGORIES = [
    "Artificial Intelligence",
    "Cyber Security",
    "Cloud Computing",
    "Robotics",
    "Data Engineering"
]

def generate_sample_dataset() -> Tuple[List[str], List[int]]:
    """
    Generates a labelled text dataset of technical abstracts and snippets
    across 5 tech domains for training the TensorFlow classifier model.
    """
    dataset = [
        # Artificial Intelligence (0)
        ("Deep neural networks and transformer architecture revolutionizing natural language processing and computer vision.", 0),
        ("Supervised learning algorithms, reinforcement learning agents, hyperparameter tuning, and gradient descent optimization.", 0),
        ("Large language models LLMs, attention mechanisms, fine-tuning embeddings, and generative AI prompt engineering.", 0),
        ("Convolutional neural networks CNNs for image segmentation, object detection, and feature extraction pipelines.", 0),
        ("Machine learning model evaluation using precision, recall, F1 score, confusion matrix, and ROC AUC curves.", 0),
        ("Backpropagation in multi-layer perceptron neural networks for classification and regression tasks.", 0),
        
        # Cyber Security (1)
        ("Network penetration testing, vulnerability scanning, threat intelligence, and zero-day exploit prevention.", 1),
        ("Public key infrastructure PKI, asymmetric encryption, RSA, AES-256 cipher, and cryptographic hash functions.", 1),
        ("Firewalls, intrusion detection systems IDS, intrusion prevention IPS, and SIEM security log analysis.", 1),
        ("Identity access management IAM, multi-factor authentication MFA, OAuth2 authorization, and zero trust security.", 1),
        ("Malware analysis, reverse engineering ransomware, rootkit detection, and incident response mitigation.", 1),
        ("Security audits, SOC2 compliance, ISO 27001 standards, and phishing email defense mechanisms.", 1),

        # Cloud Computing (2)
        ("Kubernetes container orchestration, Docker microservices deployment, ingress controllers, and auto-scaling pods.", 2),
        ("Amazon Web Services AWS EC2 instances, S3 storage buckets, CloudFront CDN, and Serverless Lambda functions.", 2),
        ("Google Cloud Platform GCP, Azure DevOps pipelines, Infrastructure as Code Terraform, and Ansible deployment.", 2),
        ("Cloud native architecture, distributed load balancing, multi-region failover, and high availability clusters.", 2),
        ("Serverless computing, event-driven architecture, API gateways, and cloud migration strategies.", 2),
        ("Virtual private cloud VPC networking, subnet routing tables, security groups, and cloud cost management.", 2),

        # Robotics (3)
        ("Autonomous mobile robots AMR, SLAM simultaneous localization and mapping, LiDAR sensor fusion, and ROS nodes.", 3),
        ("Robotic kinematics, forward and inverse dynamics, joint actuator control, and trajectory planning algorithms.", 3),
        ("Computer vision guided robotic arms, industrial automation, pick-and-place precision end effectors.", 3),
        ("Microcontroller embedded systems, ROS2 navigation stack, IMU telemetry, and motor speed control PWM.", 3),
        ("Unmanned aerial vehicles UAV drones, flight control algorithms, obstacle avoidance, and PID feedback loops.", 3),
        ("Humanoid robot locomotion, bipedal balance, impedance control, and haptic force feedback sensors.", 3),

        # Data Engineering (4)
        ("Apache Spark distributed data processing pipelines, PySpark dataframes, ETL workflows, and parquet files.", 4),
        ("Data warehousing with Snowflake, BigQuery, star schema data modeling, and OLAP analytical query processing.", 4),
        ("Real-time streaming data ingestion with Apache Kafka event streams, Flink stream processing, and topic partitions.", 4),
        ("Data lineage tracking, Apache Airflow DAG orchestration, database schema migration, and data governance.", 4),
        ("Relational SQL database indexing, B-tree query optimization, PostgreSQL partitioning, and NoSQL MongoDB document stores.", 4),
        ("Data lakehouse architecture, Delta Lake transaction logs, data quality validation, and batch processing schedules.", 4),
    ]

    # Replicate snippets to build a rich training corpus
    expanded_texts = []
    expanded_labels = []
    
    for text, label in dataset:
        for _ in range(5):
            expanded_texts.append(text)
            expanded_labels.append(label)

    return expanded_texts, expanded_labels
