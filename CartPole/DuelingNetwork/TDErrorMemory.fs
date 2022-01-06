namespace CartPole.DuelingNetwork
open CartPole.Core
open DiffSharp 
open DiffSharp.Model 
open DiffSharp.Compose
open DiffSharp.Util

type TDErrorMemory (capacity:int) = 

    let states = ResizeArray<Tensor>()
    let actions = ResizeArray<Tensor>()
    let rewards  = ResizeArray<Tensor>()
    let nextStates = ResizeArray<Tensor>()
    let isDones  = ResizeArray<Tensor>() 
    let tdErrors = ResizeArray<float>()
    let tdErrorEps = 0.0001 

    member _.Count = states.Count

    member this.Append(
        tdError     : float,
        state       : float[],
        action      : Action, 
        reward      : float, 
        nextState   : float[],
        isDone      : bool  ) =

        if this.Count > capacity then 
            tdErrors.RemoveAt 0
            states.RemoveAt 0
            actions.RemoveAt 0
            rewards.RemoveAt 0
            nextStates.RemoveAt 0
            isDones.RemoveAt 0

        tdErrors.Add    tdError
        states.Add      <| dsharp.tensor state
        actions.Add     <| dsharp.onehot (2,(action |> function | Left -> 0 | Right -> 1 ),Dtype.Int32) 
        rewards.Add     <| dsharp.tensor reward
        nextStates.Add  <| dsharp.tensor nextState
        isDones.Add     <| dsharp.tensor (isDone,Dtype.Int32)
    
    member _.Clear() = 
        tdErrors.Clear()
        states.Clear()
        actions.Clear()
        rewards.Clear()
        nextStates.Clear()
        isDones.Clear()
    
    member this.Sample (batchSize:int) =  
        let sampledIndices  =
            let indices = Random.shuffledIndices this.Count  
            [| for i in 0..batchSize-1 -> indices i |]  
        
        let stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch = 
            sampledIndices 
            |> Array.fold (fun (s,a,r,n,d ) i ->
                states.[i]      :: s,
                actions.[i]     :: a,
                rewards.[i]     :: r,
                nextStates.[i]  :: n,
                isDones.[i]     :: d ) ([],[],[],[],[])
            |> fun (s,a,r,n,d ) -> 
                (   s |> dsharp.stack,
                    a |> dsharp.stack,
                    r |> dsharp.stack |> dsharp.unsqueeze 1,
                    n |> dsharp.stack,
                    d |> dsharp.stack |> dsharp.unsqueeze 1
                )
                
        stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch

    member this.PrioritizedExperienceReplay (batchSize:int) =  
        let sampledIndices  =  
            let tdErrorSum = tdErrorEps * float this.Count + Seq.sum tdErrors
            let rands = [| for _ in 1..batchSize -> Random.Double()  * tdErrorSum |]  |> Array.sort
            let cumerror = tdErrors |> Seq.map(fun t -> t + tdErrorEps)  |> Seq.toArray |> Array.cumulativeSum  
            [|  let mutable i = 0
                for r in rands -> 
                    while i < cumerror.Length && cumerror.[i] < r do i <- i + 1
                    min i (tdErrors.Count-1) 
            |] 

        let stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch = 
            sampledIndices 
            |> Array.fold (fun (s,a,r,n,d ) i ->
                states.[i]      :: s,
                actions.[i]     :: a,
                rewards.[i]     :: r,
                nextStates.[i]  :: n,
                isDones.[i]     :: d ) ([],[],[],[],[])
            |> fun (s,a,r,n,d ) -> 
                (   s |> dsharp.stack, 
                    a |> dsharp.stack,
                    r |> dsharp.stack |> dsharp.unsqueeze 1,
                    n |> dsharp.stack,
                    d |> dsharp.stack |> dsharp.unsqueeze 1
                )
        
        stateBatch, actionBatch, rewardBatch, nextStateBatch, isDoneBatch

    member _.UpdateTDErrorMemory (policyNet:Model, targetNet:Model, discount:float) =
          
        let stateBatch,actionBatch,rewardBatch,nextStateBatch  =  
            (   states      |> dsharp.stack,
                actions     |> dsharp.stack,
                rewards     |> dsharp.stack |> dsharp.unsqueeze 1,
                nextStates  |> dsharp.stack 
            )
        policyNet.noDiff() 
        targetNet.noDiff()   
       
        let stateActionValues = 
            let am = actionBatch.argmax(1).view [-1;1]  
            (stateBatch --> policyNet).gather(1,am)  

        let expectedStateActionValue  = 
            let nextStateValues =  
                let am = (nextStateBatch --> policyNet).argmax(1).view [-1;1]    
                (nextStateBatch --> targetNet).gather(1,am)
            (rewardBatch + discount * nextStateValues ) 

        let errors = (expectedStateActionValue - stateActionValues).abs().squeeze(1).float64().toArray() :?> float[]

        tdErrors.Clear()
        tdErrors.AddRange errors 