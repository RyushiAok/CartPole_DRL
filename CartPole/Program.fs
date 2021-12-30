namespace CartPole 
open System.Runtime.InteropServices
open DiffSharp 
open DiffSharp.Backends.Torch  
open System.Threading
open System
open FSharp.Control 

module Main =   
    
    let duelingNetwork () =  
        let env = Environment(steps=200) 
        let subject = env.Subject 
        async {  
            let agent =
                DuelNetworkAgent(
                    observationSize=4,
                    hiddenSize=32, 
                    actionCount=2,
                    learningRate=0.0001,
                    discount=0.99,
                    wd= 100.0,
                    batchSize=32
                )  
            agent.Optimize(env,1000) |> ignore

            while true do 
                env.Reset()
                while not <| env.Failed() do 
                    Threading.Thread.Sleep 20
                    let action = agent.SelectAction(env.State(),0.0)
                    env.Reflect(action) |> ignore 
        } 
        |> Async.Start  
        subject

       
    let a2c () = 
        let steps = 200
        let env = Environment(steps) 
        let subject = env.Subject 
        async { 
            let actorCritic =  
                ActorCritic(
                    actor=ActorNetwork (observationSize=4,hiddenSize=32,actionCount=2),
                    critic=CriticNetwork(observationSize=4,hiddenSize=32),
                    learningRate=0.0025    // 0.01 0.003 
                )

            let agents =   
                [|yield env; for _ in 1..31 -> Environment(steps)|]
                |> Array.map(fun e -> 
                    ActorCriticAgent(
                        actorCritic = actorCritic,
                        env = e
                    )
                ) 

            A2C.optimize(actorCritic, agents) |> ignore

            let agent = ActorCriticAgent(actorCritic,env)

            while true do 
                env.Reset()
                while not <| env.Failed() do 
                    Thread.Sleep 20
                    let action = agent.SelectAction(dsharp.tensor <| [env.State()]).toInt32()
                    env.Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore 
        } 
        |> Async.Start 
        subject
        


    [<EntryPoint>]
    let main argv =  
        NativeLibrary.Load(@"C:\libtorch\lib\torch.dll") |> ignore
        dsharp.config(backend=Backend.Torch, device=Device.CPU)  
        
        let subject =  
            a2c() 

            //duelingNetwork()   

            //let subject = Environment(steps=200).Subject  
            //Observable.interval (TimeSpan.FromMilliseconds 15.0)
            //|> Observable.subscribe(fun _ -> 
            //    subject.OnNext (subject.Value.Move None ))
            //|> ignore
            //subject

        CartPole.Gui.appRun subject argv  