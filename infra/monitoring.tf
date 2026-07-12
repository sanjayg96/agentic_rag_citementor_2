# --- Phase 7: CloudWatch alarms on the Lambda's built-in metrics ---
#
# This function is rarely invoked (a demo service), so alarms are tuned for
# "any error / any near-timeout matters" rather than statistical noise
# filtering. Both alarms fire into one SNS topic with an email subscription.
# Cost: SNS topic + 2 alarms are a few cents/month while the stack is applied
# — torn down with everything else on `terraform destroy` (unlike the
# standing AWS Budget in infra/bootstrap.sh, which survives teardown).

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

# AWS requires the recipient to click a confirmation link in the first email
# before delivery actually works — there is no API to skip that step.
resource "aws_sns_topic_subscription" "alerts_email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Any Lambda invocation error in a 5-minute window. TreatMissingData is
# "notBreaching" because a rarely-queried function legitimately has long
# stretches with zero invocations (and therefore zero error-metric data).
resource "aws_cloudwatch_metric_alarm" "error_rate" {
  alarm_name          = "${local.name}-errors"
  alarm_description   = "Any citementor-api Lambda invocation error in a 5 minute window."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# p95 duration approaching the function timeout — an early warning before
# invocations start actually timing out (rather than alarming only after).
resource "aws_cloudwatch_metric_alarm" "latency_p95" {
  alarm_name          = "${local.name}-latency-p95"
  alarm_description   = "p95 Lambda duration over 80% of the ${var.lambda_timeout_s}s timeout in a 5 minute window."
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 1
  threshold           = var.lambda_timeout_s * 1000 * 0.8
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}
