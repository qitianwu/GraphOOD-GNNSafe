m_in_list=(-9 -7 -5)
m_out_list=(10 5 2 0 -1 -2 -3 -4)
lamda_list=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)
alpha_list=(0.1 0.3 0.5 0.7 0.9)
K_list=(1 2 3 5 8 12 16 24 36 48 64)
enc_list=('mlp' 'gcn' 'gat' 'gcnjk' 'mixhop' 'appnp' 'gprgnn')
dev=3

# encoder backbone architecture
for e in ${enc_list[@]}
do
  python discuss.py --dis_type backbone --method msp --backbone $e --dataset cora --ood_type structure --mode detect --use_bn --device $dev
  python discuss.py --dis_type backbone --method gnnsafe --backbone $e --dataset cora --ood_type structure --mode detect --use_bn --device $dev
  python discuss.py --dis_type backbone --method gnnsafe --backbone $e --dataset cora --ood_type structure --mode detect --use_reg --lamda 0.01 --m_in -5 --m_out -1 --use_bn --device $dev
  python discuss.py --dis_type backbone --method gnnsafe --backbone $e --dataset cora --ood_type structure --mode detect --use_prop --use_bn --device $dev
  python discuss.py --dis_type backbone --method gnnsafe --backbone $e --dataset cora --ood_type structure --mode detect --use_prop --use_reg --lamda 0.01 --m_in -5 --m_out -1 --use_bn --device $dev
done

# margin
for m_in in ${m_in_list[@]}
do
  for m_out in ${m_out_list[@]}
  do
    python discuss.py --dis_type margin --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in $m_in --m_out $m_out --device $dev
  done
done

# propagation strength and layers
for alpha in ${alpha_list[@]}
do
  for K in ${K_list[@]}
  do
    python discuss.py --dis_type prop --method gnnsafe --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --K $K --alpha $alpha --lamda 0. --device $dev
  done
done

# energy regularization weight
for lamda in ${lamda_list[@]}
do
  python discuss.py --dis_type lamda --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --lamda $lamda --device $dev
done

# time
python discuss.py --dis_type time --method msp --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --device $dev
python discuss.py --dis_type time --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --device $dev
python discuss.py --dis_type time --method gnnsafe --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --device $dev

python discuss.py --dis_type time --method msp --backbone gcn --dataset arxiv --ood_type structure --mode detect --use_bn --device $dev
python discuss.py --dis_type time --method gnnsafe --backbone gcn --dataset arxiv --ood_type structure --mode detect --use_bn --use_prop --device $dev
python discuss.py --dis_type time --method gnnsafe --backbone gcn --dataset arxiv --ood_type structure --mode detect --use_bn --use_prop --use_reg --device $dev

