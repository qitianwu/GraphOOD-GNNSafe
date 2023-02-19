gnns=('gcn' 'gat' 'mixhop' 'gcnjk' 'gatjk')
datasets=('cora' 'amazon-photo' 'coauthor-cs' 'twitch' 'arxiv')
ood_types=('feature' 'label' 'structure')
m_in_space=(-9 -7 -5)
m_out_space=(-1 -2 -3 -4)
lamda_space=(0.01 0.1 1.0)
dev=1
gnn='gcn'
data='cora'
type='feature'

for m_in in ${m_in_space[@]}
do
  for m_out in ${m_out_space[@]}
  do
    for l in ${lamda_space[@]}
    do
      python main.py --method gnnsafe --backbone $gnn --dataset $data --ood_type $type --mode detect --use_bn --use_reg --lamda $l --T 1.0 --m_in $m_in --m_out $m_out --device $dev
      python main.py --method gnnsafe --backbone $gnn --dataset $data --ood_type $type --mode detect --use_bn --use_prop --use_reg --lamda $l --T 1.0 --m_in $m_in --m_out $m_out --device $dev
    done
  done
done
