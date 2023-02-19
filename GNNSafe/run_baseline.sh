### Scripts to run baselines including ODIN, OE and Mahalanobis

# ODIN
for data in "amazon-photo" "cora" "coauthor-cs"
do
  for ood in "feature" "structure" "label"
  do
    for noise in 0.0001 0.01 0.1 0.5 1.
    do
      for T in 1. 5. 10. 20. 50. 100.
      do
        python main.py --method ODIN --backbone gcn --dataset $data --ood_type $ood --mode detect --use_bn --device $dev --noise $noise --T $T
      done
    done
  done
done

# OE
for data in "amazon-photo" "cora" "coauthor-cs"
do
  for ood in "feature" "structure" "label"
  do
      python main.py --method OE --backbone gcn --dataset $data --ood_type $ood --mode detect --use_bn --device $dev
  done
done

# Mahalanobis
for data in "amazon-photo" "cora" "coauthor-cs"
do
  for ood in "feature" "structure" "label"
  do
    python main.py --method Mahalanobis --backbone gcn --dataset $data --ood_type $ood --mode detect --use_bn --device $dev --lr 0.001 --weight_decay 0.
  done
done