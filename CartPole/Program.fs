namespace CartPole 
open System.Runtime.InteropServices
open DiffSharp 
open DiffSharp.Backends.Torch  
open System.Threading
open System
open System.Net
open System.Net.Sockets
open System.Text.RegularExpressions
open FSharp.Control 
open FSharp.Control.Reactive.Builders
open System.Text
open DiffSharp.Util
open System.Reactive.Subjects

module Main =   
    
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

       

    let a2c (env: Env) =  
         async { 
             let actorCritic =  
                 ActorCritic(
                     actor=ActorNetwork (observationSize=4,hiddenSize=32,actionCount=2),
                     critic=CriticNetwork(observationSize=4,hiddenSize=32),
                     learningRate=0.0025    // 0.01 0.003 
                 )

             let agents =   
                 [|yield env; for _ in 1..31 -> Env(env.Steps)|]
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
                     let action = agent.SelectAction(dsharp.tensor <| [env.Observations()]).toInt32()
                     env.Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore 
         }  
        


    [<EntryPoint>]
    let main argv =  
        NativeLibrary.Load(@"C:\libtorch\lib\torch.dll") |> ignore
        dsharp.config(backend=Backend.Torch, device=Device.CPU)  
        
        let fsharpEnv() = 
            let subject = 
                let env = Env(steps=200)   
                //a2c(env)  |> Async.Start 
                duelingNetwork(env)  |> Async.Start
                env.Subject 
                //let subject = Environment(steps=200).Subject  
                //Observable.interval (TimeSpan.FromMilliseconds 15.0)
                //|> Observable.subscribe(fun _ -> 
                //    subject.OnNext (subject.Value.Move None ))
                //|> ignore 
            CartPole.Gui.appRun subject argv  

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

        let pythonEnvA2C () =  
            async {
                let envs = [|
                    for i in 0..31 -> 
                        
                        let socket = new Socket(AddressFamily.InterNetwork,
                                                SocketType.Stream,
                                                ProtocolType.Tcp)

                        let endPoint = new IPEndPoint(IPAddress.Loopback, 8080 + i)
                        socket.Connect(endPoint)
                        GymEnvironment (socket)
   
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

                A2C.optimize(actorCritic, agents) |> ignore

                let agent = ActorCriticAgent(actorCritic,envs[0])

                while true do 
                    envs[0].Reset()
                    while not <| envs[0].Failed() do 
                        Thread.Sleep 20
                        let action = agent.SelectAction(dsharp.tensor <| [envs[0].Observations()]).toInt32()
                        envs[0].Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore  
            }
            |> Async.RunSynchronously
 
        //fsharpEnv() 
        // pythonEnvDuel()

        pythonEnvA2C ()

        0


