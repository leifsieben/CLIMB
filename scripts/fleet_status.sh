#!/usr/bin/env bash
# One-pass fleet audit across ALL instance states -- never just `running`.
#
# Filtering to running hides the two things that matter most. With gated self-shutdown, a STOPPED
# box is the normal end state of finished work, so stopped instances are exactly where (a) results
# that were never verified on S3 sit, and (b) reusable capacity sits. The account hits a 64 vCPU
# limit for the G bucket long before AWS runs out of hardware, so restarting a stopped box often
# succeeds when a fresh launch fails.
#
# chempfn-* boxes belong to a DIFFERENT project: reported for vCPU accounting, never touched.
set -u
KEY="${KEY:-./climb-gpu-key.pem}"
printf "%-26s %-13s %-10s %-16s %s\n" NAME TYPE STATE IP JOB
aws ec2 describe-instances \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],InstanceType,State.Name,PublicIpAddress]' \
  --output text | sort -k3,3 -k1,1 | while read -r name type state ip; do
  job="-"
  if [ "$state" = "running" ] && [ "$ip" != "None" ] && [[ "$name" != chempfn* ]]; then
    # -n: ssh must NOT read stdin, or it swallows the while-read loop's input and the
    # audit silently stops after the first reachable host -- which is exactly the kind of
    # partial-report-that-looks-complete this script exists to prevent.
    job=$(ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes -i "$KEY" \
      ec2-user@"$ip" 'pgrep -af "python (scripts/|eval_v2|finetune|pretrain)" 2>/dev/null | head -1 | sed "s#.*/##;s/ .*//"' 2>/dev/null)
    [ -z "$job" ] && job="** IDLE - no job running **"
  fi
  [ "$state" = "stopped" ] && [[ "$name" != chempfn* ]] && job="<- stopped: verify results on S3, or reuse for capacity"
  printf "%-26s %-13s %-10s %-16s %s\n" "$name" "$type" "$state" "$ip" "$job"
done
echo
aws ec2 describe-instances --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value|[0],InstanceType]' --output text |
awk '{split($2,a,"."); v=(a[2]=="xlarge"?4:(a[2]=="2xlarge"?8:(a[2]=="4xlarge"?16:(a[2]=="8xlarge"?32:(a[2]=="9xlarge"?36:0)))));
      if(substr(a[1],1,1)=="g"){ if($1 ~ /chempfn/) c+=v; else m+=v } }
     END{printf "G-bucket vCPUs: CLIMB %d + chempfn %d = %d / 64 limit  (headroom %d)\n", m, c, m+c, 64-(m+c)}'
