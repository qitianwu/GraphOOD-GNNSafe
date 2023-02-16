#Scripts to run baseline GPN and GKDE(SGCN)

### Cora
# GPN
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist" "Epist_wo_Net"
  do
    python main.py --method GPN --dataset cora --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev
  done
done


# GKDE
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist"
  do
    python main.py --method SGCN --dataset cora --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev
  done
done

### Amazon-photo
# GPN
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist" "Epist_wo_Net"
  do
    python main.py --method GPN --dataset amazon-photo --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev --epochs 1000
  done
done

#GKDE
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist"
  do
    python main.py --method SGCN --dataset amazon-photo --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev --epochs 1000
  done
done

### Coauthor
# GPN
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist" "Epist_wo_Net"
  do
    python main.py --method GPN --dataset coauthor-cs --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev
  done
done

# GKDE
for oodtype in "feature" "structure" "label"
do
  for detect_type in "Alea" "Epist"
  do
    python main.py --method SGCN --dataset coauthor-cs --ood_type $oodtype --mode detect --GPN_detect_type $detect_type --device $dev
  done
done

### Twitch
# GPN
for detect_type in "Alea" "Epist"
  do
    python main.py --method GPN --dataset twitch --mode detect --GPN_detect_type $detect_type --device $dev
  done

# GKDE
for detect_type in "Alea" "Epist"
  do
    python main.py --method SGCN --dataset twitch --mode detect --GPN_detect_type $detect_type --device $dev
  done

### Arxiv
# GPN
for detect_type in "Alea" "Epist"
  do
    python main.py --method GPN --dataset arxiv --mode detect --GPN_detect_type $detect_type --device $dev
  done

# GKDE
for detect_type in "Alea" "Epist"
  do
    python main.py --method SGCN --dataset arxiv --mode detect --GPN_detect_type $detect_type --device $dev
  done