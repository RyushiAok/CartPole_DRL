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
 

type Actor(
    network:QNetwork,
    env:Env,
    actionCount     : int, 
    discount        : float,
    eps : float 

) =
     
    //let wd = defaultArg wd 300.0 // 100 -> 100.0, 
    //let batchSize = defaultArg batchSize 32
    //let memory    = Replay(1000) 

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

    //let optimizer = Adam(network,lr=dsharp.tensor learningRate) 
    //member a.a = 0

    let mutable cnt = 0
    
    member _.Log = log.ToArray()

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
        let mutable totalEpsodeRewards = 0.0
        for i in 0..buf.Length-1 do 
            let obs = env.Observations()
            let act = this.SelectAction(obs)
            let nxtObs, reward, isDone = env.Reflect act
            totalEpsodeRewards <- totalEpsodeRewards + reward
            obss[i] <- dsharp.tensor obs 
            acts[i] <- dsharp.onehot (2,(act |> function | Left -> 0 | _ -> 1 ),Dtype.Int32) 
            rewards[i] <- dsharp.tensor reward
            nxtObss[i] <- dsharp.tensor nxtObs
            isDones[i] <- dsharp.tensor (if isDone then 1 else 0)
            cnt <- cnt+ 1
            if isDone then 
                log.Add(sw.Elapsed.TotalSeconds, cnt)
                cnt <- 0

                env.Reset() 
                //resetCnt <- resetCnt + 1
                //printfn "reset %A" resetCnt
                totalEpsodeRewards <- 0.0
                ()
            else
                ()

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

