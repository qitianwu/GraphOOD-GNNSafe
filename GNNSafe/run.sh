### Cora with structure ood

python main.py --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1

### Cora with feature ood

python main.py --method msp --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.01 --device 1

### Cora with label ood

python main.py --method msp --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_reg --m_in -5 --m_out -4 --lamda 1.0 --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -4 --lamda 1.0 --device 1


### Amazon-photo with structure ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_reg --m_in -9 --m_out -1 --lamda 0.01 --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -1 --lamda 0.1 --device 1


### Amazon-photo with feature ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 1.0 --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -1 --lamda 1.0 --device 1


### Amazon-photo with label ood

python main.py --method msp --backbone gcn --dataset amazon-photo --ood_type label --mode detect --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_reg --m_in -9 --m_out -4 --lamda 1.0 --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -4 --lamda 1.0 --device 1


### Coauthor with structure ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device 1


### Coauthor with feature ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_reg --m_in -7 --m_out -1 --lamda 0.1 --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -7 --m_out -1 --lamda 0.1 --device 1


### Coauthor with label ood

python main.py --method msp --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device 1


### Twitch

python main.py --method msp --backbone gcn --dataset twitch --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_reg --m_in -7 --m_out -2 --lamda 0.1 --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -1 --lamda 0.1 --device 1

### Arxiv

python main.py --method msp --backbone gcn --dataset arxiv --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_reg --m_in -9 --m_out -4 --lamda 0.01 --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --device 1
python main.py --method gnnsafe --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -2 --lamda 0.01 --device 1

