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

        let pythonEnv () = 
            let client = 
                let socket = 
                    let endPoint = IPEndPoint(IPAddress.Any, 8080)
                    let socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp)
                    socket.Bind(endPoint)
                    socket.Listen(10)
                    socket 
                socket.Accept() 
            let env = GymEnvironment(client)  
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

        let pythonEnv2 () =  
            async {
                let envs = [|
                    for i in 0..15 ->
                        let client = 
                            let socket = 
                                let endPoint = IPEndPoint(IPAddress.Any, 8080+i)
                                let socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp) 
                                socket.Bind(endPoint)
                                socket.Listen(10)
                                socket 
                            socket.Accept() 
                        GymEnvironment(client)   
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
            

        //let pythonEnv2 () = 
        //    //async { 
        //        let client = 
        //            let socket = 
        //                let endPoint = IPEndPoint(IPAddress.Any, 8080)
        //                let socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp)
        //                socket.Bind(endPoint)
        //                socket.Listen(10)
        //                socket 
        //            socket.Accept() 
        //        let steps = 200 
        //        let clientMessage i = 
        //            //printfn "%A" <| sprintf "^\%d+" i
        //            let regexId = Regex(sprintf "^%d" i, RegexOptions.Compiled)
        //            let regexObs = Regex("^\d+,o,(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled) 
        //            let regexAct = Regex("^\d+,r,(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)$", RegexOptions.Compiled)   
        //            let (|Obs|Act|Non|)  (input:string) =
        //                if regexId.Match(input).Success then 
        //                    regexObs.Match(input) |> fun m -> 
        //                        if m.Success then 
        //                            Obs ( 
        //                                m.Groups.Values 
        //                                |> Seq.toArrayQuick 
        //                                |> fun a -> a.[1..]
        //                                |> Array.map(fun t -> t.ToString() |> System.Double.Parse) 
        //                            )
        //                        else 
        //                            regexAct.Match(input) |> fun m -> 
        //                                if m.Success then 
        //                                    Act (
        //                                        m.Groups.Values 
        //                                        |> Seq.toArrayQuick 
        //                                        |> fun a -> a.[1..]
        //                                        |> Array.map(fun t -> t.ToString() |> System.Double.Parse)
        //                                        |> fun a -> a.[0..3], a[4..] 
        //                                    )
        //                                else 
        //                                    Non
        //                else 
        //                    Non 
        //            let clientMsg = 
        //                let buffer = Array.create 512 (byte 0) 
        //                let rec loop () =  
        //                    asyncSeq {
        //                        let len = client.Receive(buffer) 
        //                        let msg = Encoding.Default.GetString(buffer, 0, len)
        //                        printfn "recv:%s" msg
        //                        match msg with 
        //                        | Obs obs -> 
        //                            yield (ClientMessage.Observations obs )
        //                            yield! loop ()
        //                        | Act (obs,res) -> 
        //                            yield (ClientMessage.ActionResult (obs, res[0], res[1]=1.))
        //                            yield! loop ()
        //                        | Non -> yield! loop ()  
        //                    }  
        //                loop () 
        //            clientMsg 
        //        let actorCritic =  
        //            ActorCritic(
        //                actor=ActorNetwork (observationSize=4,hiddenSize=32,actionCount=2),
        //                critic=CriticNetwork(observationSize=4,hiddenSize=32),
        //                learningRate=0.0025    // 0.01 0.003 
        //            ) 
        //        let envs = 
        //            [| for i in 0..1 ->  
        //                let cm = clientMessage i  
        //                //cm
        //                //|> AsyncSeq.subscribe(fun msg -> 
        //                //    printfn "%A" msg
        //                //)
        //                //|> ignore
        //                let cc = BehaviorSubject ClientCommand.Reset
        //                cc 
        //                |> Observable.subscribe (fun cmd -> 
        //                     match cmd with 
        //                     | ClientCommand.Reset ->
        //                         sprintf "%d,reset" i 
        //                         |> fun t -> printfn "%s" t ; t
        //                         |> Encoding.Default.GetBytes 
        //                         |> client.Send
        //                         |> ignore
        //                     | ClientCommand.Action str ->
        //                        sprintf "%d,%s" i str 
        //                        |> fun t -> printfn "%s" t ; t
        //                        |> Encoding.Default.GetBytes 
        //                        |> client.Send
        //                        |> ignore 
        //                     //printfn "%d,act" i
        //                )
        //                |> ignore
        //                GymEnvironment'(
        //                    cm,
        //                    cc,
        //                    steps) 
        //            |] 
        //        let agents =
        //            envs  
        //            |> Array.map(fun e -> 
        //                ActorCriticAgent(
        //                    actorCritic = actorCritic,
        //                    env = e
        //                )
        //            ) 
        //        A2C.optimize(actorCritic, agents) |> ignore 
        //        let agent = ActorCriticAgent(actorCritic,envs[0]) 
        //        while true do 
        //            envs[0].Reset()
        //            while not <| envs[0].Failed() do 
        //                Thread.Sleep 20
        //                let action = agent.SelectAction(dsharp.tensor <| [envs[0].Observations()]).toInt32()
        //                envs[0].Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore 
            //} |> Async.RunSynchronously
        //fsharpEnv() 
        //pythonEnv()

        pythonEnv2 ()

        0


