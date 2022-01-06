namespace CartPole.A2C
open CartPole.Core
open DiffSharp
open DiffSharp.Model
open DiffSharp.Compose
open DiffSharp.Optim

type ActorNetwork(observationSize: int, hiddenSize: int, actionCount: int) =
    inherit Model() 
    let fc1 = Linear(observationSize,hiddenSize) 
    let fc2 = Linear(hiddenSize,hiddenSize)  
    let fc3 = Linear(hiddenSize,actionCount) 

    do
        base.add [fc1;fc2;fc3]

    override _.forward (input:Tensor) =
        input  
        --> fc1
        --> dsharp.tanh
        --> fc2
        --> dsharp.tanh
        --> fc3 


type CriticNetwork(observationSize: int, hiddenSize: int) =
    inherit Model() 
    let fc1 = Linear(observationSize,hiddenSize) 
    let fc2 = Linear(hiddenSize,hiddenSize)  
    let fc3 = Linear(hiddenSize,1) 

    do
        base.add [fc1;fc2;fc3]

    override _.forward (input:Tensor) =
        input 
        --> fc1
        --> dsharp.relu
        --> fc2
        --> dsharp.relu
        --> fc3  

type ActorCritic(actor:ActorNetwork, critic:CriticNetwork,learningRate:float) =  

    let actorOptimizer  = Adam(actor, lr=dsharp.tensor learningRate)
    let criticOptimizer = Adam(critic,lr=dsharp.tensor learningRate)
     
    let entropy (input:Tensor , actions: Tensor) = 
        let probs    = input --> actor |> dsharp.softmax 1
        let logProbs = input --> actor |> dsharp.logsoftmax 1 |> dsharp.mul actions
        let entropy = -(logProbs * probs) |> dsharp.sum -1 |> dsharp.mean
        logProbs,entropy  

    member _.Value (input : Tensor) =
        input --> critic  
         
    member _.Act (state : Tensor)  =  
        state --> actor |> dsharp.softmax 1 |> dsharp.multinomial 1 


    member this.Update (states:Tensor, actions:Tensor, actionValues:Tensor, agents:int, advanced:int) = 
        actor.noDiff()
        critic.noDiff()
        actor.reverseDiff() 
        critic.reverseDiff()  

        let logProbs, entropy  = entropy(states,actions) 
        let logProbs = logProbs |> dsharp.view[advanced;agents;2]  

        let advantages = 
            let values   = this.Value states |> dsharp.view [advanced;agents;1]  
            actionValues.[0..4] - values 

        let actionGain = (logProbs*advantages.clone()).mean()  


        let loss = 
            let criticLoss = advantages.pow(2).mean()
            criticLoss * 0.5 - actionGain - entropy * 0.001 
        
        loss.reverse()

        // clip_grad_normに対応する関数? 
        // https://github.com/pytorch/pytorch/blob/e858f6eed9ca8417027557667ca4fb1ed9250321/torch/nn/utils/clip_grad.py 
        // nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm) 

        actorOptimizer.step() 
        criticOptimizer.step()  


        ()
         
 


     
