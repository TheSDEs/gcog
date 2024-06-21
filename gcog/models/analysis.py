import torch
import src.task.constants as constants

def accuracy_score(network,outputs,targets,averaged=True):
    """
    return accuracy given a set of outputs and targets
    """
    acc = [] # accuracy array
    max_resp = torch.argmax(outputs,1)
    if outputs.dim()==2:
        max_resp = torch.argmax(outputs,1)
    else:
        max_resp = torch.argmax(outputs)
    acc = max_resp==targets
    acc = acc.float()
    if averaged:
        acc = torch.mean(acc)

    return acc



def forward_transformer_analysis(model, rule_inputs, stim_inputs, noise=False, dropout=False):
    """
    this forward pass is for analysis of models only, not for training
    #
    Run a forward pass of a trial by input_elements matrix
    For each time window, pass each 
    rule_inputs (Tensor): batch x seq_length x rule_vec
    stim_inputs (Tensor): batch x seq_length x stim_vec x time
    """
    assert(rule_inputs.device==stim_inputs.device)
    device = rule_inputs.device
    #Add noise to inputs
    if noise:
        rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
        stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

    with torch.no_grad():
        # output lists
        all_embedding = []
        all_attn_out = []
        all_transformer_out = []
        all_hn = []
        all_outputs = []
        ####
        # Rule input first
        inputs = torch.concat((rule_inputs,
                            torch.zeros(stim_inputs[:,:,:,0].size(),device=device)),dim=-1)
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(model.num_rnn_layers,
                        batch_size,
                        model.num_hidden, 
                        device=device)
        # transformer block
        embedding = model.w_embed(inputs)
        attn_outputs, attn_out_weights = model.selfattention(embedding, embedding, embedding, need_weights=False)
        attn_outputs = model.layernorm0(attn_outputs)
        transformer_out = model.mlp(attn_outputs)
        transformer_out = model.layernorm1(transformer_out)
        # rnn
        outputs, hn = model.rnn.forward(transformer_out,hn) 

        all_embedding.append(embedding)
        all_attn_out.append(attn_outputs)
        all_transformer_out.append(transformer_out)
        all_hn.append(hn)
        all_outputs.append(outputs)

        ####
        # Stim input 2nd
        n_time = len(constants.ALL_TIME)
        # first make sure rule and target are only included in the last time step
        for t in range(n_time):
            # Input to hidden
            inputs = torch.concat((torch.zeros(rule_inputs.size(),device=device),
                                    stim_inputs[:,:,:,t]),dim=-1)
            # transformer block
            embedding = model.w_embed(inputs)
            attn_outputs, attn_out_weights = model.selfattention(embedding, embedding, embedding, need_weights=False)
            attn_outputs = model.layernorm0(attn_outputs)
            transformer_out = model.mlp(attn_outputs)
            transformer_out = model.layernorm1(transformer_out)
            # rnn 
            outputs, hn = model.rnn.forward(transformer_out,hn) 
            outputs = model.layernorm2(outputs)
            outputs = model.w_out(hn[-1]) #1 x batch x output
            # store
            all_embedding.append(embedding)
            all_attn_out.append(attn_outputs)
            all_transformer_out.append(transformer_out)
            all_hn.append(hn)
            all_outputs.append(outputs)

        return all_outputs, all_hn, all_attn_out, all_transformer_out # only want the last output from seq data


def forward_rnn_analysis(model, rule_inputs, stim_inputs, noise=False,dropout=False):
    """
    this forward pass is for analysis of models only, not for training
    #
    Run a forward pass of a trial by input_elements matrix
    For each time window, pass each 
    rule_inputs (Tensor): batch x seq_length x rule_vec
    stim_inputs (Tensor): batch x seq_length x stim_vec x time
    """
    assert(rule_inputs.device==stim_inputs.device)
    device = rule_inputs.device

    with torch.no_grad():
        all_hn = []
        all_outputs = []
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        ####
        # Rule input first
        inputs = torch.concat((rule_inputs,
                            torch.zeros(stim_inputs[:,:,:,0].size(),device=device)),dim=-1)
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(model.num_hidden_layers,
                        batch_size,
                        model.num_hidden, 
                        device=device)

        outputs, hn = model.forward(inputs,hn) 
        hn = model.layernorm(hn)
        # store
        all_hn.append(hn)
        all_outputs.append(outputs)

        ####
        # Stim input 2nd
        n_time = len(constants.ALL_TIME)
        # first make sure rule and target are only included in the last time step
        for t in range(n_time):
            # Input to hidden
            inputs = torch.concat((torch.zeros(rule_inputs.size(),device=device),
                                    stim_inputs[:,:,:,t]),dim=-1)
            outputs, hn = model.forward(inputs,hn) 
            hn = model.layernorm(hn)
            outputs = torch.squeeze(model.layernorm(outputs))
            #outputs = model.w_out(hn[-1]) #1 x batch x output
            outputs = model.w_out(outputs) #1 x batch x output
            # store
            all_hn.append(hn)
            all_outputs.append(outputs)
        return all_outputs, all_hn # only want the last output from seq data

def forward_lstm_analysis(model, rule_inputs, stim_inputs, noise=False, dropout=False):
    """
    this forward pass is for analysis of models only, not for training
    #
    Run a forward pass of a trial by input_elements matrix
    For each time window, pass each 
    rule_inputs (Tensor): batch x seq_length x rule_vec
    stim_inputs (Tensor): batch x seq_length x stim_vec x time
    """
    assert(rule_inputs.device==stim_inputs.device)
    device = rule_inputs.device

    with torch.no_grad():
        all_hn = []
        all_cn = []
        all_outputs = []
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5


        ####
        # Rule input first
        inputs = torch.concat((rule_inputs,
                                torch.zeros(stim_inputs[:,:,:,0].size(),device=device)),dim=-1)
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(model.num_hidden_layers,
                            batch_size,
                            model.num_hidden, 
                            device=device)
        cn = torch.zeros(model.num_hidden_layers,
                            batch_size,
                            model.num_hidden, 
                            device=device)

        outputs, (hn, cn) = model.forward(inputs,(hn,cn)) 
        # store
        all_hn.append(hn)
        all_cn.append(cn)
        all_outputs.append(outputs)

        ####
        # Stim input 2nd
        n_time = len(constants.ALL_TIME)
        # first make sure rule and target are only included in the last time step
        for t in range(n_time):
            # Input to hidden
            inputs = torch.concat((torch.zeros(rule_inputs.size(),device=device),
                                    stim_inputs[:,:,:,t]),dim=-1)
            outputs, (hn, cn) = model.forward(inputs,(hn,cn)) 
            #outputs = model.layernorm(outputs)
            hn = model.layernorm(hn)
            outputs = model.w_out(hn[-1]) #1 x batch x output
            #outputs = model.w_out(outputs) #1 x batch x output
            # store
            all_hn.append(hn)
            all_cn.append(cn)
            all_outputs.append(outputs)

        return all_outputs, all_hn, all_cn # only want the last output from seq data
