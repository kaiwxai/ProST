from pre_train import EGCN
from utils import *
from torch import nn, optim
import torch
from copy import deepcopy
from ProST.utils import seed, seed_everything
from random import shuffle
from ProST.meta import MAML
from ProST.eva import acc_f1_over_batches
import random
import pickle as pk
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import numpy as np
from torchviz import make_dot
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from ProST.prompt import GNN, FrontAndHead
import logging

# from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(filename='./log/multi_task.log', level=logging.INFO)
logger = logging.getLogger()

handler = logging.FileHandler('./logmulti_task.log')
logger.addHandler(handler)


device = torch.device("cuda:" + str(3)) if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
logging.info(device)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


seed = 3407
np.random.seed(seed)
random.seed(seed)
seed_everything(seed)
batch_size = 32
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
def model_components(meta_lr):
    """
    input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'

    :param args:
    :param round:
    :param pre_train_path:
    :param gnn_type:
    :param project_head_path:
    :return:
    """
    adapt_lr = 0.001
    # meta_lr = 0.01
    model = FrontAndHead(batch_size=batch_size, input_dim=100, hid_dim=100, num_classes=2,  # 0 or 1
                         task_type="multi_label_classification",
                         token_num=10, cross_prune=0.1, inner_prune=0.3)
    model = model.to(device)
    # load pre-trained GNN
    feats_per_node = 100
    layer_1_feats = 100
    layer_2_feats = 100
    feats = [feats_per_node, layer_1_feats, layer_2_feats]

    gnn = EGCN(feats, device)
    for p in gnn.parameters():
        p.requires_grad = False
    
    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)
    lossfn = nn.CrossEntropyLoss(reduction='mean')
    lossmse = nn.MSELoss()

    return maml, opt, lossfn, lossmse, gnn


def induced_graph_2_K_shot(t1_dic, t2_dic, flag, task_index, dataname: str = None, K=None, seed=None):
    if dataname is None:
        raise KeyError("dataname is None!")
    if K:
        t1_pos = t1_dic
        t2_pos = t2_dic  # treat as neg
    else:
        t1_pos = t1_dic
        t2_pos = t2_dic      # treat as neg
    # print('--------------------')
    
    task_data = []
    time_range_length = 12

    for key, value in t1_pos.items():
        # print(len(value))
        for g in value:
            g.y = torch.tensor([1]).long()
        # task_data.append(value[-1])        
        for i in range(len(value) - time_range_length + 1):
            task_data.append(value[i:i+1][-1])

    for key, value in t2_pos.items():
        for g in value:
            g.y = torch.tensor([0]).long()
        # task_data.append(value[-1])
        for i in range(len(value) - time_range_length + 1):
            task_data.append(value[i:i+1][-1])
    
    if flag:
        if task_index == 'od':
            num_samples = int(len(task_data)*0.1)
        else:
            num_samples = int(len(task_data)*0.1)
        # num_samples = int(len(task_data))
        task_data = random.sample(task_data, num_samples)

    print(len(task_data))
    task_data = MyDataset(task_data)
    task_data = DataLoader(task_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return task_data


def load_tasks(task_index: str, meta_stage: str, task_pairs: list, index: int, dataname: str = None, K_shot=None, seed=0):
    if dataname is None:
        raise KeyError("dataname is None!")
    task_1, task_2 = task_pairs[index]
    task_1_support = './test_data/{}/{}.{}.{}.support'.format(task_index, dataname, task_1, meta_stage)
    task_1_query = './test_data/{}/{}.{}.{}.query'.format(task_index, dataname, task_1, meta_stage)
    task_2_support = './test_data/{}/{}.{}.{}.support'.format(task_index, dataname, task_2, meta_stage)
    task_2_query = './test_data/{}/{}.{}.{}.query'.format(task_index, dataname, task_2, meta_stage)

    logging.info(task_1_support)
    logging.info(task_1_query)

    logging.info(task_2_support)
    logging.info(task_2_query)

    with open(task_1_support, 'br') as t1s, open(task_1_query, 'br') as t1q, open(task_2_support, 'br') as t2s, open(task_2_query, 'br') as t2q:
        t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
        flag = True
        support = induced_graph_2_K_shot(t1s_dic, t2s_dic, flag, task_index, dataname, K=K_shot, seed=seed)
        flag = False
        t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
        query = induced_graph_2_K_shot(t1q_dic, t2q_dic, flag, task_index, dataname, K=K_shot, seed=seed)

    return support, query, len(task_pairs)
    

def meta_train_maml(epoch, task_index, maml, gnn, lossfn, lossmse, opt, meta_train_task_id_list, dataname, lr, adapt_steps=2, K_shot=100):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)
    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]
    # torch.backends.cudnn.enabled = False
    # meta-training
    if task_index == 'node':
        flow_matrix = np.load('./output/{}_count.npy'.format(dataname, dataname))
        logging.info(flow_matrix.shape)
    elif task_index == 'od':
        flow_matrix = np.load('./output/{}_flow.npy'.format(dataname, dataname))
    elif task_index == 'subgraph':
        flow_matrix = np.load('./test_data/{}/{}_flow_matrix.npy'.format(task_index, dataname, dataname), allow_pickle=True)

    mean = np.mean(flow_matrix)
    std = np.std(flow_matrix)
    flow_matrix = (flow_matrix - mean) / std
    flow_matrix = torch.tensor(flow_matrix).to(device)
    node_feature = 0
    node_seq = 0

    support1, query1, total_num1 = load_tasks(task_index, 'train', task_pairs, 0, dataname, seed)
    support2, query2, total_num1 = load_tasks(task_index, 'train', task_pairs, 1, dataname, seed)
    
    for ep in range(K_shot):
        logging.info('--------')
        meta_train_loss = 0.0
        mape_list = []
        rmse_list = []
        mae_list = []
        for support_batch1, query_batch1, support_batch2, query_batch2 in zip(support1, query1, support2, query2):
            learner = maml.clone()
            for _ in range(2):  # adaptation_steps
                for support_batch in [support_batch1, support_batch2]:
                    support_loss = 0.
                    support_batch = support_batch.to(device)
                    support_batch_preds, support_batch_y, support_flow_value, actual_value= learner(support_batch, 
                                                                                                    flow_matrix,
                                                                                                    node_feature, 
                                                                                                    node_seq,
                                                                                                    gnn, 
                                                                                                    task_index,
                                                                                                    device)
                    support_mse_loss = lossmse(support_flow_value, actual_value)
                    support_batch_loss = lossfn(support_batch_preds, support_batch_y)
                    support_loss = support_loss + support_batch_loss + support_mse_loss
                    support_loss = support_loss / len(support_batch)
                    
                    learner.adapt(support_loss)
            
            query_loss = 0.  
            for query_batch in [query_batch1, query_batch2]:
                query_batch = query_batch.to(device)
                query_batch_preds, query_batch_y, query_flow_value, actual_value = learner(query_batch,
                                                                                            flow_matrix,
                                                                                            node_feature, 
                                                                                            node_seq,
                                                                                            gnn,
                                                                                            task_index, 
                                                                                            device)
                query_batch_loss = lossfn(query_batch_preds, query_batch_y)
                query_mse_loss = lossmse(query_flow_value, actual_value)
                rmse = compute_rmse(std * query_flow_value + mean, std * actual_value + mean)
                mae = compute_mae(std * query_flow_value + mean, std * actual_value + mean)
                mape = calculate_mape(std * query_flow_value + mean, std * actual_value + mean)
                rmse_list.append(rmse)
                mae_list.append(mae)
                mape_list.append(mape)
                query_loss = query_loss + query_batch_loss + query_mse_loss

            query_loss = query_loss / 2
            meta_train_loss += query_loss
            
            opt.zero_grad()
            query_loss.backward()
            opt.step()
        logging.info((sum(rmse_list)/len(rmse_list)))
        logging.info((sum(mae_list)/len(mae_list)))
        logging.info(meta_train_loss/(batch_size*2))
        with open("output_5e-3_73_0113_{}".format(dataname) + str(task_index) + ".txt", "a", encoding="utf-8") as file:  
            file.write(str(sum(rmse_list)/len(rmse_list)) + "\n")
            file.write(str(sum(mae_list)/len(mae_list)) + "\n")
            file.write(str(meta_train_loss) + "\n")
        torch.save(maml, './pre_train_model_dict/maml_multi_task73_poi.pth')

def meta_test_adam(meta_test_task_id_list,
                   task_index,
                   dataname,
                   seed,
                   maml, 
                   gnn,
                   adapt_steps_meta_test,
                   lossfn,
                   type_dataset):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)
    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]
    logging.info(task_pairs)
    test_model = deepcopy(maml.module)
    test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                          lr=0.0001, weight_decay=0.000001) # )

    test_model.train()
    if task_index == 'node':
        flow_matrix = np.load('./output/{}_count.npy'.format(dataname, dataname))
    elif task_index == 'od':
        flow_matrix = np.load('./output/{}_flow.npy'.format(dataname, dataname))
    elif task_index == 'subgraph':
        flow_matrix = np.load('./test_data/{}/{}_flow_matrix.npy'.format(task_index, dataname, dataname), allow_pickle=True)

    mean = np.mean(flow_matrix)
    std = np.std(flow_matrix)
    flow_matrix = (flow_matrix - mean) / std
    flow_matrix = torch.tensor(flow_matrix).to(device)
    node_feature = 0
    node_seq = 0
    graph_emb_list = []
    type_dataset_index = type_dataset[0]
    support1, query1, total_num1 = load_tasks(task_index, type_dataset_index, task_pairs, 0, dataname, seed)
    support2, query2, total_num1 = load_tasks(task_index, type_dataset_index, task_pairs, 1, dataname, seed)
    for _ in range(adapt_steps_meta_test):
        mape_list = []
        rmse_list = []
        mae_list = []
        for support_batch1, query_batch1, support_batch2, query_batch2 in zip(support1, query1, support2, query2):
            for support_batch in [support_batch1, support_batch2]:
                support_batch = support_batch.to(device)
                support_loss = 0.
                support_batch_preds, support_batch_y, support_flow_value, actual_value = test_model(support_batch,
                                            flow_matrix,
                                            node_feature, 
                                            node_seq,
                                            gnn, 
                                            task_index,
                                            device)
                support_batch_loss = lossfn(support_batch_preds, support_batch_y)
                support_mse_loss = lossmse(support_flow_value, actual_value)

                support_loss = support_loss + support_batch_loss + support_mse_loss
                support_loss = support_loss / len(support_batch)
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()
    
        
        test_model.eval()
        for support_batch1, query_batch1, support_batch2, query_batch2 in zip(support1, query1, support2, query2):
            for query_batch in [query_batch1, query_batch2]:
                query_batch = query_batch.to(device)
                query_batch_preds, query_batch_y, query_flow_value, actual_value = test_model(query_batch,
                                                                                            flow_matrix,
                                                                                            node_feature, 
                                                                                            node_seq,
                                                                                            gnn, 
                                                                                            task_index,
                                                                                            device)
                graph_emb_list.append(graph_emb.cpu().detach())
                rmse = compute_rmse(std * query_flow_value + mean, std * actual_value + mean)
                mae = compute_mae(std * query_flow_value + mean, std * actual_value + mean)
                mape = calculate_mape(std * query_flow_value + mean, std * actual_value + mean)
                rmse_list.append(rmse)
                mae_list.append(mae)
                mape_list.append(mape)
        logging.info('---------------------')
        logging.info((sum(mae_list)/len(mae_list)))
        logging.info((sum(rmse_list)/len(rmse_list)))
        with open("output_5e-3_73_test_0123_pre_{}".format(dataname) + str(task_index) + ".txt", "a", encoding="utf-8") as file:  
            file.write(str(sum(rmse_list)/len(rmse_list)) + "\n")
            file.write(str(sum(mae_list)/len(mae_list)) + "\n")
            file.write('-------------' + "\n")
        print('------')

if __name__ == '__main__':
    adapt_steps_meta_test = 50  # 00  # 50
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    maml, opt, lossfn, lossmse, gnn = model_components(args.lr)
   
    maml = torch.load('./pre_train_model_dict/maml_multi_task73_poi.pth')
    print(maml)
    task = ['node', 'od', 'subgraph']  # ,
    # task = ['od', 'node', 'subgraph'] 
    meta_train_task_id_list = [0, 1, 2, 3]
    meta_test_task_id_list = [0, 1, 2, 3]

    # dataname = 'Manhattan'
    # for task_index in task:
    #     params_to_freeze_node = maml.pre_flow_node.parameters()
    #     params_to_freeze_od = maml.pre_flow_od.parameters()
    #     params_to_freeze_subgraph = maml.pre_flow_subgraph.parameters()
    #     if task_index == 'node':
    #         for param in params_to_freeze_node:
    #             param.requires_grad = False
    #         for param in params_to_freeze_od:
    #             param.requires_grad = True
    #         for param in params_to_freeze_subgraph:
    #             param.requires_grad = True
    #     if task_index == 'od':
    #         for param in params_to_freeze_node:
    #             param.requires_grad = True
    #         for param in params_to_freeze_od:
    #             param.requires_grad = False
    #         for param in params_to_freeze_subgraph:
    #             param.requires_grad = True
    #     if task_index == 'subgraph':
    #         for param in params_to_freeze_node:
    #             param.requires_grad = True
    #         for param in params_to_freeze_od:
    #             param.requires_grad = True
    #         for param in params_to_freeze_subgraph:
    #             param.requires_grad = False

    #     meta_train_maml(20, task_index, maml, gnn, lossfn, lossmse, opt, meta_train_task_id_list,
    #                     dataname, args.lr, adapt_steps = 2, K_shot=50)
       
    dataname = 'Brooklyn'
    for task_index in task:
        params_to_freeze_node = maml.pre_flow_node.parameters()
        params_to_freeze_od = maml.pre_flow_od.parameters()
        params_to_freeze_subgraph = maml.pre_flow_subgraph.parameters()
        if task_index == 'node':
            for param in params_to_freeze_node:
                param.requires_grad = False
            for param in params_to_freeze_od:
                param.requires_grad = True
            for param in params_to_freeze_subgraph:
                param.requires_grad = True
        if task_index == 'od':
            for param in params_to_freeze_node:
                param.requires_grad = True
            for param in params_to_freeze_od:
                param.requires_grad = False
            for param in params_to_freeze_subgraph:
                param.requires_grad = True
        if task_index == 'subgraph':
            for param in params_to_freeze_node:
                param.requires_grad = True
            for param in params_to_freeze_od:
                param.requires_grad = True
            for param in params_to_freeze_subgraph:
                param.requires_grad = False
        # meta_test_adam(meta_test_task_id_list, task_index, dataname, seed, maml, gnn, adapt_steps_meta_test, lossfn)
        type_dataset = ['train', 'test']
        meta_test_adam(meta_test_task_id_list, task_index, dataname, seed, maml, gnn, adapt_steps_meta_test, lossfn, type_dataset)


    
