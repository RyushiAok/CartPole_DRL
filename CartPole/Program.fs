namespace CartPole 

open CartPole.Core 

open System.Runtime.InteropServices
open DiffSharp 
open DiffSharp.Backends.Torch  
open System.Threading
open System
open System.Net
open System.Net.Sockets
open System.Text.RegularExpressions
open FSharp.Control 
open FSharp.Control.Reactive
open FSharp.Control.Reactive.Builders
open System.Text
open DiffSharp.Util
open System.Reactive.Subjects


module Main =   
    
    module DuelNet = 
        open CartPole.DuelingNetwork 
        let duelingNetwork () =    
            let env = Env(steps=200)  
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
                        let action = agent.SelectAction(env.Observations(),0.0)
                        env.Reflect(action) |> ignore 
            } |> Async.RunSynchronously   
            env.Subject
    
        let pythonEnvDuel () =  
            let socket = new Socket(AddressFamily.InterNetwork,
                                    SocketType.Stream,
                                    ProtocolType.Tcp)

            let endPoint = new IPEndPoint(IPAddress.Loopback, 8080 )
            socket.Connect(endPoint)
            let env = GymEnvironment (socket)
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
                        let action = agent.SelectAction(env.Observations(),0.0)
                        env.Reflect(action) |> ignore 
            } |> Async.RunSynchronously

    module A2C = 
        open CartPole.A2C

        let a2c () =  
             let env = Env(steps=200)  
             async { 
                 let actorCritic =  
                     ActorCritic(
                         actor=ActorNetwork (observationSize=4,hiddenSize=32,actionCount=2),
                         critic=CriticNetwork(observationSize=4,hiddenSize=32),
                         learningRate=0.0025    // 0.01 0.003 
                     )

                 let agents =   
                     [|yield env; for _ in 1..60 -> Env(env.Steps)|]
                     |> Array.map(fun e -> 
                         ActorCriticAgent(
                             actorCritic = actorCritic,
                             env = e
                         )
                     ) 

                 let sw = System.Diagnostics.Stopwatch()
                 sw.Start()
                 A2C.optimize(actorCritic, agents) |> ignore
                 sw.Elapsed |> printfn "\n%A"

                 let agent = ActorCriticAgent(actorCritic,env)

                 while true do 
                     env.Reset()
                     while not <| env.Failed() do 
                         Thread.Sleep 20
                         let action = agent.SelectAction(dsharp.tensor <| [env.Observations()]).toInt32()
                         env.Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore 
             } |> Async.RunSynchronously
             env.Subject


        let pythonEnvA2C () =  
            async {
                let envs = [|
                    for i in 0..59 ->  
                        let socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp) 
                        let endPoint = new IPEndPoint(IPAddress.Loopback, 8080 + i)
                        socket.Connect(endPoint)
                        GymEnvironment (socket, steps=200) 
                |]
            
                let actorCritic =  
                    ActorCritic(
                        actor=ActorNetwork (observationSize=4,hiddenSize=32,actionCount=2),
                        critic=CriticNetwork(observationSize=4,hiddenSize=32),
                        learningRate=0.0025    // 0.01 0.003 
                    )

                let agents =   
                    envs
                    |> Array.map(fun e -> 
                        ActorCriticAgent(
                            actorCritic = actorCritic,
                            env = e
                        )
                    ) 

                let sw = System.Diagnostics.Stopwatch()
                sw.Start()
                A2C.optimize(actorCritic, agents) |> ignore
                sw.Elapsed |> printfn "\n%A"

                let agent = ActorCriticAgent(actorCritic,envs[0])

                while true do 
                    envs[0].Reset()
                    while not <| envs[0].Failed() do 
                        Thread.Sleep 20
                        let action = agent.SelectAction(dsharp.tensor <| [envs[0].Observations()]).toInt32()
                        envs[0].Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore  
            }
            |> Async.RunSynchronously


    module ApeXDQN = 
        open CartPole.ApeXDQN 
        let apeXdqn () =  
            let env = Env(steps=500) 
            let model = 
                let path = sprintf @"%A\%A" __SOURCE_DIRECTORY__ @"Model\apeXdqn2_20220110080143.pth"
                Model.Model.load(path)
                //QNetwork(observationSize=4,hiddenSize=64,actionCount=2)
            async { 
                let n = 16
                let envs = [|yield env; for _ in 1..n-1 -> Env(env.Steps)|] 
                let actors = 
                    let actorNetworks =  
                        Array.init n (fun _ ->model.clone() )
                    Array.zip actorNetworks envs
                    |> Array.mapi(fun i (net,env) -> Actor(net, env, 2, 0.98,  (float i / float n) * 0.4 + 0.01)  ) // 0.7
                let learnerNetwork = model
                let learner = Learner(learnerNetwork, actors, 0.98, 0.001)  
                learner.Learn(1, 50)   
                learner.Save(sprintf "apeXdqn_%s.pth" <| DateTime.Now.ToString("yyyyMMddHHmmss"))
                let agent =  Actor(learnerNetwork, envs[0], 2, 0.98, 0)
                while true do 
                    envs[0].Reset()
                    while not <| envs[0].Failed() do 
                        Thread.Sleep 20
                        let action = agent.SelectAction( envs[0].Observations()) 
                        envs[0].Reflect(action ) |> ignore  
            }  
            |> Async.Start
            env.Subject

        
        let apeXdqn2 () =  
            let steps = 600
            let env = Env2(steps) 
            let env' = Env2(steps) 
            let model =  
                let path = sprintf @"%A\%A" __SOURCE_DIRECTORY__ @"Model\apeXdqn2_20220110080143.pth"
                Model.Model.load(path) 
                //QNetwork(observationSize=4,hiddenSize=64,actionCount=2)
            async { 
                let n = 32
                let envs = [|yield env; for _ in 1..n-1 -> Env2(steps)|] 
                let actors = 
                    let actorNetworks = Array.init n (fun _ -> model.clone() )
                    Array.zip actorNetworks envs
                    |> Array.mapi(fun i (net,env) -> Actor(net, env, 2, 0.98,  (float i / float n) * 0.8 + 0.01 ) ) // 0.7
                let learnerNetwork = model
                let learner = Learner(learnerNetwork, actors, 0.98, 0.001)   
                async{
                    learner.Learn(500, 50)   
                    learner.Save(sprintf "apeXdqn2_%s.pth" <| DateTime.Now.ToString("yyyyMMddHHmmss")) 
                } |> Async.Start
                
                let agent =  Actor(model.clone(), env' , 2, 0.98, 0)
                while true do 
                    env'.Reset()
                    agent.UpdateParam(model.parameters)
                    while not <| env'.IsDone() do  
                        Thread.Sleep 20 
                        env'.Observations()
                        |> agent.SelectAction
                        |> env'.Reflect
                        |> ignore
            }  
            |> Async.Start
            //env.Subject
            env'.Subject

    [<EntryPoint>]
    let main argv =  
        NativeLibrary.Load(@"C:\libtorch\lib\torch.dll") |> ignore 
        dsharp.config(backend=Backend.Torch, device=Device.CPU)  
        
        //CartPole.Gui.appRun (DuelNet.duelingNetwork()) argv |> ignore
        //CartPole.Gui.appRun (A2C.a2c()) argv  |> ignore
        //CartPole.Gui.appRun (ApeXDQN.apeXdqn()) argv  |> ignore
        CartPole.Gui.appRun (ApeXDQN.apeXdqn2()) argv |> ignore 
        //pythonEnvDuel() 
        //pythonEnvA2C () 
        0


