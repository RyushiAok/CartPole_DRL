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
                //let am = states.argmax(1).view [-1;1]  
                //(states --> globalNet).gather(1,am)  
                (states --> globalNet).mul(acts)

            let TQ  = 
                let nextStateValues =  
                    let am = (nextStates --> globalNet).argmax(1).view [-1;1]    
                    (nextStates --> globalNet).gather(1,am)
                (rewards + discount * (1 - isDones) * nextStateValues ).mul(acts)  

            let tdErrors = (TQ - Q)**2
            let loss =
                //dsharp.mean (tdErrors.mul (dsharp.tensor weights))
                //dsharp.mean ( (dsharp.tensor weights).expand([weights.Length; 2]).mul tdErrors
                dsharp.mean ( (dsharp.tensor (weights  |> Array.map(fun t -> [t;t]))).mul tdErrors)
            loss.reverse()
            optimizer.step() 
            globalNet.noDiff()
            
            indicesAll.AddRange(indices) 
            tdErrorsAll.AddRange( 
                let am = tdErrors.argmax(1).view([-1;1])
                //printfn"%A" <| tdErrors.gather(1,am).flatten()
                tdErrors.gather(1,am).flatten().toArray() :?> float32[] |> Array.map float
            ) 
        (globalNet.parameters.copy(), indicesAll.ToArray(), tdErrorsAll.ToArray())

    member this.Learn()= 
        let replay = Replay(bufferSize= (1<<<14)) 
             
        let a () =
            let sw = System.Diagnostics.Stopwatch()
            sw.Start()
            let learning = 
                actors
                |> Array.toList // ?
                |> List.map(fun actor -> 
                    asyncSeq{ 
                        while true do   
                            yield actor.RollOut(globalNet.parameters) // 重みを更新 
                    }
                )
                |> AsyncSeq.mergeAll   
            learning
            |> AsyncSeq.take 30
            |> AsyncSeq.iter(fun (tdError, (obss, acts, nxtObss, rewards, isDones)) -> 
                replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )
            ) 
            |> Async.RunSynchronously

            // minibatch取得、更新
            let mutable minibatchs = [| for _ in 0..15 -> replay.SampleMinibatch(batchSize=32)|]
            //this.UpdateNetwork(minibatchs) // これは非同期にできるらしい
            let mutable i = 0
             
            learning
            |> AsyncSeq.take 300 
            |> AsyncSeq.iterAsync  (fun (tdError, (obss, acts, nxtObss, rewards, isDones)) -> 
                async { 
                    printfn "%A %A" sw.Elapsed i
                    i <- i + 1
                    replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )
                    let networkParams ,indices, tdErrors = this.UpdateNetwork(minibatchs) // this.UpdateNetworkの終了を待機
                    // networkParamsを各Actorに送信
                    replay.UpdatePriority(indices, tdErrors) |> ignore
                    minibatchs <-  [| for _ in 0..15 -> replay.SampleMinibatch(batchSize=32)|]  
                }
            )
            |> Async.RunSynchronously
             
    
            () 

        
        let b () =
            let sw = System.Diagnostics.Stopwatch()
            let lockObj = obj()
            let learning = 
                actors
                |> Array.toList // ?
                |> List.map(fun actor -> 
                    asyncSeq{ 
                        while true do   
                            yield  actor.RollOut(globalNet.parameters.copy() ) // 重みを更新
                    }
                )
                |> AsyncSeq.mergeAll  

            learning
            |> AsyncSeq.take 30
            |> AsyncSeq.iter(fun (tdError, (obss, acts, nxtObss, rewards, isDones)) -> 
                replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )
            ) 
            |> Async.RunSynchronously

            // minibatch取得、更新
            let mutable minibatchs = [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|]
            //this.UpdateNetwork(minibatchs) // これは非同期にできるらしい 
            learning
            |> AsyncSeq.iter (fun (tdError, (obss, acts, nxtObss, rewards, isDones)) -> 
                lock lockObj (fun _ -> 
                    replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )  
                )
            )
            |> Async.Start  
            for i in 0..1000 do 
                printf  "%A" i   
                sw.Restart()
                let weights ,indices, tdErrors = this.UpdateNetwork(minibatchs) // this.UpdateNetworkの終了を待機
                
                lock lockObj (fun _ ->     
                    printf "\tn %A"  sw.Elapsed; sw.Restart()
                    replay.UpdatePriority(indices, tdErrors) |> ignore
                    //printf "\tp %A"  sw.Elapsed; sw.Restart()
                    minibatchs <-  [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|] 
                    //printf "\tm %A"  sw.Elapsed; sw.Restart() 
                )
                // replay.Add(tdError, obss, acts, nxtObss, rewards, isDones )  
                //replay.UpdatePriority(indices, tdErrors) |> ignore
                //minibatchs <-  [| for _ in 0..15 -> replay.SampleMinibatch(batchSize=32)|]  
                printfn ""

 
        let c () =
            let lockObj = obj()
            let mutable now = 1
            let tas =
                actors 
                |> Array.toList
                |> List.map(fun actor -> 
                    asyncSeq  {
                        let mutable prev = -1
                        while true do 
                            let (tdError, (obss, acts, nxtObss, rewards, isDones)) =
                                while now = prev do 
                                    ()
                                prev <- now
                                actor.RollOut(globalNet.parameters)
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
            |> AsyncSeq.iter(fun _ -> now <- -now  )
            |> Async.RunSynchronously     
            let mutable minibatchs = [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|]
            tas
            |> AsyncSeq.take 3000
            |> AsyncSeq.iteriAsync(fun i _ -> 
                async { 
                    printfn "%A" i
                    if i % actors.Length = 0 then  
                        let weights ,indices, tdErrors = this.UpdateNetwork(minibatchs) 
                        now <- now + 1
                        replay.UpdatePriority(indices, tdErrors) |> ignore
                        minibatchs <-  [| for _ in 0..actors.Length-1 -> replay.SampleMinibatch(batchSize=32)|] 
                        ()
                } 
            )
            |> Async.RunSynchronously
            |> ignore

             
            Chart.Line(xy = actors[0].Log )
            |> Chart.show
            ()
    
         //a()  
        //b()
        c()
         
        ()