namespace CartPole.ApeXDQN
// https://horomary.hatenablog.com/entry/2021/03/02/235512#%E5%88%86%E6%95%A3%E5%AD%A6%E7%BF%92%E3%81%AE%E5%AE%9F%E8%A3%85
open CartPole.Core
open DiffSharp 
open DiffSharp.Model  
open DiffSharp.Compose    
open DiffSharp.Util
open DiffSharp.Optim
            
type QNetwork (observationSize: int, hiddenSize: int, actionCount: int) = 
    inherit Model()  
    let fc1     = Linear(observationSize,hiddenSize) 
    let fc2     = Linear(hiddenSize,hiddenSize) 
    let fc3Adv  = Linear(hiddenSize,actionCount)
    let fc3V    = Linear(hiddenSize,1)

    do 
        base.add([ fc1; fc2; fc3Adv; fc3V])  

    override _.forward x =
        let h1  = x  --> fc1 --> dsharp.relu   
        let h2  = h1 --> fc2 --> dsharp.relu    
        let adv = h2 --> fc3Adv
        let v   = h2 --> fc3V |> dsharp.expand [-1;adv.shape.[1]]  
        v + adv - adv.mean(1,keepDim=true).expand[-1;adv.shape.[1] ]    
 


type QNetwork2 (observationSize: int, hiddenSize: int, actionCount: int) = 
    inherit Model()  
    let fc1     = Linear(observationSize,hiddenSize) 
    let fc2     = Linear(hiddenSize,hiddenSize) --> dsharp.relu -->Linear(hiddenSize,hiddenSize) --> dsharp.relu --> Linear(hiddenSize,hiddenSize)
    let fc3Adv  = Linear(hiddenSize,actionCount)
    let fc3V    = Linear(hiddenSize,1)

    do 
        base.add([ fc1; fc2; fc3Adv; fc3V])  

    override _.forward x =
        let h1  = x  --> fc1 --> dsharp.relu   
        let h2  = h1 --> fc2 --> dsharp.relu    
        let adv = h2 --> fc3Adv
        let v   = h2 --> fc3V |> dsharp.expand [-1;adv.shape.[1]]  
        v + adv - adv.mean(1,keepDim=true).expand[-1;adv.shape.[1] ]    
 

type Actor(
    network:Model,
    env:Environment,
    actionCount     : int, 
    discount        : float,
    eps : float 

) = 
    let buf = Array.zeroCreate 100
    let log = ResizeArray()
    let sw = System.Diagnostics.Stopwatch()
    do 
        sw.Start()
    let obss, acts, nxtObss, rewards, isDones = 
        Array.zeroCreate 100,
        Array.zeroCreate 100,
        Array.zeroCreate 100,
        Array.zeroCreate 100,
        Array.zeroCreate 100 
         

    let mutable cnt = 0
    let mutable totalEpisodeRewards = 0.0
    
    member _.Log = log 

    member _.UpdateParam(networkParameters:ParameterDict) =  
        network.parameters <- networkParameters.copy()

    member _.SelectAction (state: float[]) = 
        if Random.Double() < eps then 
            Random.Integer(0,actionCount) 
            |> function 
                | 0 -> Left | _ -> Right
        else  
            network.noDiff() 
            (dsharp.tensor [state] --> network).argmax(1).[0].toInt32() 
            |> function 
                | 0 -> Left | _ -> Right

    member this.RollOut(networkParameters:ParameterDict) = 
        network.parameters <- networkParameters.copy()
        for i in 0..buf.Length-1 do 
            let obs = env.Observations()
            let act = this.SelectAction(obs)
            let nxtObs, reward, isDone = env.Reflect act
            totalEpisodeRewards <- totalEpisodeRewards + reward
            obss[i] <- dsharp.tensor obs 
            acts[i] <- dsharp.onehot (2,(act |> function | Left -> 0 | _ -> 1 ),Dtype.Int32) 
            rewards[i] <- dsharp.tensor reward
            nxtObss[i] <- dsharp.tensor nxtObs
            isDones[i] <- dsharp.tensor (if isDone then 1 else 0)
            cnt <- cnt+ 1
            if isDone then   
                //log.Add(sw.Elapsed.TotalSeconds, cnt)
                log.Add(sw.Elapsed.TotalSeconds, totalEpisodeRewards)
                cnt <- 0 
                env.Reset()  
                totalEpisodeRewards <- 0.0 

        let obsBatch, actBatch, rewardBatch, nxtObsBatch, isDonesBatch = 
            (   obss     |> dsharp.stack,
                acts     |> dsharp.stack,
                rewards  |> dsharp.stack |> dsharp.unsqueeze 1,
                nxtObss  |> dsharp.stack,
                isDones  |> dsharp.stack |> dsharp.unsqueeze 1
            )
        
        let Q = 
            let am = actBatch.argmax(1).view [-1;1]  
            (obsBatch --> network).gather(1,am)  

        let TQ  = 
            let nextStateValues =  
                let am = (nxtObsBatch --> network).argmax(1).view [-1;1]    
                (nxtObsBatch --> network).gather(1,am)
            (rewardBatch + discount * (1 - isDonesBatch) * nextStateValues ) 

        let tdErrors = (TQ-Q).abs().squeeze(1).float64().toArray() :?> float[]
        tdErrors, (obss, acts, nxtObss, rewards, isDones)

