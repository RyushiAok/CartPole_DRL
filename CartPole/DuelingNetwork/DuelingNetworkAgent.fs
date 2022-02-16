namespace CartPole.DuelingNetwork
open CartPole.Core
open DiffSharp  
open DiffSharp.Optim
open DiffSharp.Util 
open DiffSharp.Compose
open Plotly.NET 

type DuelNetworkAgent(
    observationSize : int,
    hiddenSize      : int,
    actionCount     : int,
    learningRate    : float,
    discount        : float,
    ?wd : float,
    ?batchSize : int) =
     
    let wd = defaultArg wd 300.0 // 100 -> 100.0, 
    let batchSize = defaultArg batchSize 32
    let memory    = TDErrorMemory(capacity=10000)
    let policyNet = DuelingNetwork(observationSize,hiddenSize,actionCount)
    let targetNet = DuelingNetwork(observationSize,hiddenSize,actionCount) 
    let optimizer = Adam(policyNet,lr=dsharp.tensor learningRate) 

    member _.SelectAction (state: float[], eps: float) = 
        if Random.Double() < eps then 
            Random.Integer(0,actionCount) 
            |> function 
                | 0 -> Left | _ -> Right
        else  
            policyNet.noDiff() 
            (dsharp.tensor [state] --> policyNet).argmax(1).[0].toInt32() 
            |> function 
                | 0 -> Left | _ -> Right

    member this.Step (env:Environment, episode:int, batchSize:int) = 
        let state  = env.Observations()
        let action = this.SelectAction  (state, 0.01 + ( 0.9- 0.01) * exp( -1.0 * float episode / wd)) 

        let newState,reward,isDone = env.Update action

        memory.Append(
            tdError     = 0.0,
            state       = state,
            action      = action,
            reward      = reward,
            nextState   = newState,
            isDone      = isDone
        )    
         
        if memory.Count >= batchSize then 
            let stateBatch,actionBatch,rewardBatch,nextStateBatch, _  =
                if episode < 30 then  
                    memory.Sample(batchSize)  
                else 
                    memory.PrioritizedExperienceReplay(batchSize) 
            policyNet.reverseDiff()  
            
            let stateActionValues = 
                (stateBatch --> policyNet).mul(actionBatch)  

            let expectedStateActionValue  = 
                let nextStateValues =  
                    let am = (nextStateBatch --> policyNet).argmax(1) |> dsharp.view [-1;1]    
                    (nextStateBatch --> targetNet).gather(1,am)
                (rewardBatch + discount * nextStateValues ).mul(actionBatch) 

            let loss = dsharp.smoothL1Loss(stateActionValues,expectedStateActionValue)  
            loss.reverse()
            optimizer.step() 
            loss.toDouble()
        else
            0.0  

    member _.UpdateTarget() =   
        targetNet.parameters <- policyNet.parameters.copy() 

    member this.Optimize(env: Environment, episodes) = 
        
        let logStep = ResizeArray() 
        let logLoss = ResizeArray()
        let returnsLast10episodes = Array.create 100 0.0 
        let aveRet = ResizeArray()
         
        let rec loop episode =  
            env.Reset()   
            if Array.average returnsLast10episodes > float (env.Steps-5) || episode > episodes then 
                episode - 1
            else  
                printf  "episode %d | " episode                        
                let sumLoss = 
                    let rec steps acc =
                        if env.IsDone() 
                        then acc 
                        else steps (acc + this.Step(env,episode,batchSize=batchSize))
                    steps 0.0 

                memory.UpdateTDErrorMemory(policyNet,targetNet,discount) 

                if episode % 5 = 0 then this.UpdateTarget()  
                 
                logStep.Add(logStep.Count,env.Elappsed())
                logLoss.Add(logLoss.Count,sumLoss / float(env.Elappsed())) 
                returnsLast10episodes.[episode % returnsLast10episodes.Length] <- float <| env.Elappsed()
                aveRet.Add( aveRet.Count,Array.average returnsLast10episodes) 
                printfn "ave %4.1f, record %3d, loss: %A, "
                    (Array.average returnsLast10episodes) (env.Elappsed()) (sumLoss / float( env.Elappsed()) ) 

                loop (episode+1)  

        let e =  loop 1 

        [
            Chart.combine [ 
                Chart.Scatter(logStep,StyleParam.Mode.Markers) 
                Chart.Line(aveRet)
            ]
            |> Chart.withLegend false
            |> Chart.withYAxisStyle "record"

            Chart.Line(logLoss )  
            |> Chart.withLegend false
            |> Chart.withYAxisStyle "loss"
        ] 
        |> Chart.SingleStack(Pattern= StyleParam.LayoutGridPattern.Coupled)
        |> Chart.withLayoutGridStyle(YGap= 0.1)
        |> Chart.withXAxisStyle "episode" 
        |> Chart.withTitle (sprintf  "wd = %A" wd)
        |> Chart.show 

        e
