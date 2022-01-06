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
        let duelingNetwork (env: Env) =   
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
            }   
    
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

        let a2c (env: Env) =   
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
             }


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
            let env = Env(steps=200) 
            async { 
                let envs = [|yield env; for _ in 1..20 -> Env(env.Steps)|] 
                let actorNetworks = 
                    [| for i in 0..20 -> QNetwork(observationSize=4,hiddenSize=32,actionCount=2) |]
                let actors = 
                    Array.zip actorNetworks envs
                    |> Array.mapi(fun i (net,e) -> Actor(net, e, 2, 0.98, (float i / 20.0) * 0.5) )
                let learnerNetwork =  QNetwork(observationSize=4,hiddenSize=32,actionCount=2)
                let lerner = Learner(learnerNetwork, actors, 0.98, 0.0025)//0.003
                lerner.Learn() 

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

    [<EntryPoint>]
    let main argv =  
        NativeLibrary.Load(@"C:\libtorch\lib\torch.dll") |> ignore 
        dsharp.config(backend=Backend.Torch, device=Device.CPU)  
        
        let fsharpEnv() = 
            let subject = 
                let env = Env(steps=200)   
                let s = env.Subject
                //a2c(env)  |> Async.Start 
                //duelingNetwork(env)  |> Async.Start
                //let subject = Environment(steps=200).Subject  
                //Observable.interval (TimeSpan.FromMilliseconds 15.0)
                //|> Observable.subscribe(fun _ -> 
                //    subject.OnNext (subject.Value.Move None ))
                //|> ignore 
                s
            CartPole.Gui.appRun subject argv  
             

 
        // fsharpEnv() 
        //pythonEnvDuel()

        //pythonEnvA2C ()
        

        //let s = ApeXDQN.apeXdqn() 
        CartPole.Gui.appRun (ApeXDQN.apeXdqn()) argv  
      
  
        0


