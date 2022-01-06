namespace CartPole.ApeXDQN

open DiffSharp
open DiffSharp.Model
open DiffSharp.Compose
open DiffSharp.Optim
 
open FSharp.Control
open FSharp.Control.Reactive
open FSharp.Control.Reactive.Builders

open Plotly.NET
 
type Learner(globalNet:QNetwork, actors:Actor[], discount:float, learningRate:float)=  
    let optimizer = Adam(globalNet,lr=dsharp.tensor learningRate) 
    member _.UpdateNetwork(minibatchs: (int[]* float[]*Transitions)[]) = // globalNetの学習 
        let indicesAll = ResizeArray<int>()
        let tdErrorsAll = ResizeArray<float>()
        for indices, weights, transitions in minibatchs do 
            let states = transitions.states     |> dsharp.stack 
            let acts = transitions.actions     |> dsharp.stack 
            let rewards = transitions.rewards  |> dsharp.stack |> dsharp.unsqueeze 1 
            let nextStates = transitions.nextStates  |> dsharp.stack 
            let isDones = transitions.isDones  |> dsharp.stack|> dsharp.unsqueeze 1 
            
            globalNet.reverseDiff()
            let Q =   
                (states --> globalNet).mul(acts)

            let TQ  = 
                let nextStateValues =  
                    let am = (nextStates --> globalNet).argmax(1).view [-1;1]    
                    (nextStates --> globalNet).gather(1,am)
                (rewards + discount * (1 - isDones) * nextStateValues ).mul(acts)  

            let tdErrors = (TQ - Q)**2
            let loss = 
                dsharp.mean ( (dsharp.tensor (weights  |> Array.map(fun t -> [t;t]))).mul tdErrors)
            loss.reverse()
            optimizer.step() 
            globalNet.noDiff()
            
            indicesAll.AddRange(indices) 
            tdErrorsAll.AddRange( 
                let am = tdErrors.argmax(1).view([-1;1]) 
                tdErrors.gather(1,am).flatten().toArray() :?> float32[] |> Array.map float
            ) 
        ( indicesAll.ToArray(), tdErrorsAll.ToArray())


    member this.Learn()= 
        let replay = Replay(bufferSize= (1<<<14))  
        let lockObj = obj() // ?
        let tas =
            actors 
            |> Array.toList
            |> List.map(fun actor -> 
                asyncSeq  { 
                    while true do 
                        let (tdError, (obss, acts, nxtObss, rewards, isDones)) = 
                            actor.RollOut(globalNet.parameters)
                        //replay.Add(tdError, obss, acts, nxtObss, rewards, isDones ) 
                        lock lockObj (fun _ -> 
                            replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )  
                        )
                        yield 0
                        ()
                    ()
                } 
            ) 
            |> AsyncSeq.mergeAll
            
        tas
        |> AsyncSeq.take 30  
        |> AsyncSeq.iter(fun _ -> () ) //now <- -now  )
        |> Async.RunSynchronously     
        let mutable minibatchs =
            [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|] 

        tas
        |> AsyncSeq.take 2000
        |> AsyncSeq.iteriAsync(fun i _ -> 
            async { 
                printfn "%A" i
                if i % actors.Length = 0 then  
                    let indices, tdErrors = this.UpdateNetwork(minibatchs)  
                    replay.UpdatePriority(indices, tdErrors) 
                    minibatchs <-  
                        [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|] 
                    //lock lockObj (fun _ -> 
                    //    replay.UpdatePriority(indices, tdErrors) 
                    //    minibatchs <-  
                    //        [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|] 
                    //)
                    ()
            } 
        )
        |> Async.RunSynchronously
        |> ignore


             
        Chart.Line(xy = actors[0].Log )
        |> Chart.show
        ()
      
        ()