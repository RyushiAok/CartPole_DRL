namespace CartPole.ApeXDQN
open System.Threading.Tasks
open FSharp.Control.TaskBuilder
open DiffSharp
open DiffSharp.Model
open DiffSharp.Compose
open DiffSharp.Optim
open Plotly.NET
 
type Learner(globalNet:Model, actors:Actor[], discount:float, learningRate:float)=  
    let optimizer = Adam(globalNet,lr=dsharp.tensor learningRate) 
    member _.UpdateNetwork(minibatchs: (int[]* float[]*Transitions)[]) =
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

    member this.Learn(thres, n, name)=  
        let replay = Replay(bufferSize= (1<<<14))  
        let miniN = 16 
        let rollOut i =  
            //task {  
            //    let (tdError, (obss, acts, nxtObss, rewards, isDones)) = 
            //        actors[i].RollOut(globalNet.parameters) 
            //    return (tdError, (obss, acts, nxtObss, rewards, isDones))  
            //} :> Task
            async {  
                let (tdError, (obss, acts, nxtObss, rewards, isDones)) = 
                    actors[i].RollOut(globalNet.parameters) 
                return (tdError, (obss, acts, nxtObss, rewards, isDones))  
            } 
            |> Async.StartAsTask 
            :> Task

        let rollOuts = actors |> Array.mapi(fun i _ -> rollOut i) 
                 
        for _ in 0..30 do 
            // https://stackoverflow.com/questions/5116712/task-waitall-on-a-list-in-f
            let id = Task.WaitAny rollOuts
            let (tdError, (obss, acts, nxtObss, rewards, isDones)) = 
                rollOuts[id] 
                :?> Task<float[] * (Tensor[]*Tensor[]*Tensor[]*Tensor[]*Tensor[])>
                |> fun t -> t.Result
            replay.Add (tdError, obss, acts, nxtObss, rewards, isDones)  
            rollOuts[id] <- rollOut id
            
        let learn (replay: Replay) = 
            //task { 
            //    let minibatchs = Array.init miniN (fun _ -> replay.SampleMinibatch(batchSize=32)) 
            //    let indices, tdErrors = this.UpdateNetwork(minibatchs)  
            //    return indices, tdErrors
            //}

            
            async { 
                let minibatchs = Array.init miniN (fun _ -> replay.SampleMinibatch(batchSize=32)) 
                let indices, tdErrors = this.UpdateNetwork(minibatchs)  
                return indices, tdErrors
            }
            |> Async.StartAsTask    
            
        let mutable learner = learn replay 
        let mutable updateCnt, i, prev = 0, 0, actors[0].Log.Count  
        let records = Array.create n 0.0  
        let select = fst // fst: step , snd: reward
        while Array.average records < thres do
            if i % 1000 = 999 then 
                let name = sprintf "%s_%s.pth" name (System.DateTime.Now.ToString("yyyyMMddHHmmss"))  
                globalNet.save(sprintf @"%s\Model\%s" __SOURCE_DIRECTORY__  name)

            
            System.Console.CursorLeft <- 0
            // printf "%A" [for actor in actors -> actor.Elapsed ]

            if prev <> actors[0].Log.Count then
                prev <- actors[0].Log.Count 
                let elappsed, record =
                    let (e, (step, reward)) = actors[0].Log[actors[0].Log.Count-1]  
                    e, select (step, reward)
                records[i % records.Length] <- record |> float
                i <- i + 1  
                //System.Console.CursorLeft <- 0
                printfn  "%5d | time %.1f | record %4.1f | avg %4.1f\t " i elappsed record (Array.average records) 
                actors[1..] |> Array.iter(fun actor -> actor.Log.Clear())

            let id = Task.WaitAny rollOuts
            let (tdError, (obss, acts, nxtObss, rewards, isDones)) = 
                rollOuts[id] 
                :?> Task<float[] * (Tensor[]*Tensor[]*Tensor[]*Tensor[]*Tensor[])>
                |> fun t -> t.Result
            replay.Add (tdError, obss, acts, nxtObss, rewards, isDones)  
            rollOuts[id] <- rollOut id 
            if learner.IsCompleted then  
                replay.UpdatePriority learner.Result
                updateCnt <- updateCnt + 1  
                learner <- learn replay
                  
        Chart.Line(
            x = (actors[0].Log |> Seq.unzip |> fst ),
            y = (actors[0].Log |> Seq.unzip |> snd |> Seq.map select ) 
        )
        |> Chart.show 