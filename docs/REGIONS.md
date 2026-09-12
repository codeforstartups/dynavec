# Amazon S3 Vectors Regional Availability & Support Matrix

`dynavec` combines **Amazon DynamoDB** with **Amazon S3 Vectors** to deliver serverless, low-cost, ultra-low-latency vector storage in your own AWS account.

DynamoDB is available across all AWS commercial and sovereign regions. S3 Vectors is natively supported in the AWS regions listed below.

---

## Regional Availability Matrix

| Geographic Area | AWS Region Name | Region Code | S3 Vectors Support | DynamoDB Support |
| :--- | :--- | :--- | :---: | :---: |
| **North America** | US East (N. Virginia) | `us-east-1` | ✅ Supported | ✅ Supported |
| | US East (Ohio) | `us-east-2` | ✅ Supported | ✅ Supported |
| | US West (Oregon) | `us-west-2` | ✅ Supported | ✅ Supported |
| | US West (N. California) | `us-west-1` | ✅ Supported | ✅ Supported |
| | Canada (Central) | `ca-central-1` | ✅ Supported | ✅ Supported |
| | Canada West (Calgary) | `ca-west-1` | ✅ Supported | ✅ Supported |
| | Mexico (Central) | `mx-central-1` | ✅ Supported | ✅ Supported |
| **Europe** | Europe (Frankfurt) | `eu-central-1` | ✅ Supported | ✅ Supported |
| | Europe (Ireland) | `eu-west-1` | ✅ Supported | ✅ Supported |
| | Europe (London) | `eu-west-2` | ✅ Supported | ✅ Supported |
| | Europe (Paris) | `eu-west-3` | ✅ Supported | ✅ Supported |
| | Europe (Stockholm) | `eu-north-1` | ✅ Supported | ✅ Supported |
| | Europe (Milan) | `eu-south-1` | ✅ Supported | ✅ Supported |
| | Europe (Spain) | `eu-south-2` | ✅ Supported | ✅ Supported |
| | Europe (Zurich) | `eu-central-2` | ✅ Supported | ✅ Supported |
| **Asia Pacific** | Asia Pacific (Tokyo) | `ap-northeast-1` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Seoul) | `ap-northeast-2` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Osaka) | `ap-northeast-3` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Mumbai) | `ap-south-1` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Hyderabad) | `ap-south-2` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Singapore) | `ap-southeast-1` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Sydney) | `ap-southeast-2` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Melbourne) | `ap-southeast-4` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Jakarta) | `ap-southeast-3` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Malaysia) | `ap-southeast-5` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Thailand) | `ap-southeast-7` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Hong Kong) | `ap-east-1` | ✅ Supported | ✅ Supported |
| | Asia Pacific (Taipei) | `ap-east-2` | ✅ Supported | ✅ Supported |
| | Asia Pacific (New Zealand) | `ap-southeast-6` | ✅ Supported | ✅ Supported |
| **South America** | South America (São Paulo) | `sa-east-1` | ✅ Supported | ✅ Supported |
| **Africa** | Africa (Cape Town) | `af-south-1` | ✅ Supported | ✅ Supported |
| **AWS GovCloud** | AWS GovCloud (US-East) | `us-gov-east-1` | ✅ Supported | ✅ Supported |
| | AWS GovCloud (US-West) | `us-gov-west-1` | ✅ Supported | ✅ Supported |

---

## Configuring Region in `dynavec`

You can configure the target AWS region in several ways:

### 1. Via `DynavecConfig`
```python
from dynavec import Dynavec, DynavecConfig

config = DynavecConfig(
    vector_bucket="my-vector-bucket",
    index="my-vector-index",
    table="my-dynamo-table",
    dimension=768,
    region="us-east-1",  # or any supported region code
)
db = Dynavec(config=config)
```

### 2. Via Standard AWS Environment Variables
If `region_name` is omitted in code, `dynavec` automatically reads your AWS environment variables:
```bash
export AWS_DEFAULT_REGION=us-east-1
# or
export AWS_REGION=us-east-1
```

---

## Official References

* [AWS Regional Services Availability](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/)
* [Amazon S3 Documentation](https://docs.aws.amazon.com/s3/)
* [Amazon DynamoDB Documentation](https://docs.aws.amazon.com/dynamodb/)
