namespace CartPole.A2C
open CartPole.Core
 
open DiffSharp
open DiffSharp.Compose
open DiffSharp.Util

open Plotly.NET  

type ActorCriticAgent(actorCritic: ActorCritic, env: Environment) = 

    let l = ResizeArray()

    member _.logSteps = l

    member _.Observe ()  = env.Observations()

    member _.Step (action:int) =   
        let newState,reward,isDone = env.Update (action |> function | 0 -> Left | _ -> Right)
        if isDone then
            l.Add(l.Count, env.Elappsed())
            env.Reset()
        newState, action, reward,isDone 

    member _.SelectAction state = state |> actorCritic.Act 


type A2C =

    static member optimize (actorCritic: ActorCritic, agents: ActorCriticAgent[], ?adv: int, ?discount: float, ?iters: int) = 
        let adv = defaultArg adv 5  
        let discount = defaultArg discount 0.9
        let iters = defaultArg iters 3000

        for i in 1..iters do   
            System.Console.CursorLeft <- 0
            printf "%A/ %A " i iters

            let states ,actions, rewards, isDones = 
                let rec advance n states actions rewards isDones = 
                    if n = adv then 
                        states  |> List.rev, 
                        actions |> List.rev,
                        rewards |> List.rev,
                        isDones |> List.rev
                    else
                        let acts = 
                            let observations = 
                                agents
                                |> Array.map (fun a -> a.Observe())
                                |> dsharp.tensor  
                            (observations --> actorCritic.Act |> dsharp.squeeze).toArray() :?> int[]   
                        agents 
                        |> Array.mapi(fun i agent -> async { return agent.Step acts.[i] } )
                        |> Async.Parallel
                        |> Async.RunSynchronously   
                        |> fun res ->
                            Array.foldBack(fun (xa,xb,xc,xd) (a,b,c,d) -> 
                                ( xa::a, 
                                  dsharp.onehot (2, xb,Dtype.Int32)::b, 
                                  [xc]::c,
                                  [if xd then 0.0 else 1.0]::d ) ) res ([],[],[],[])  
                        |> fun (s, a, r, d) ->  
                            advance 
                                (n+1) 
                                (dsharp.tensor s::states) 
                                (dsharp.stack a::actions) 
                                (dsharp.tensor r::rewards) 
                                (dsharp.tensor d::isDones) 
                             
                advance 0 [agents |> Array.map (fun a -> a.Observe()) |> dsharp.tensor] [] [] [] 

            let actionValues =  
                List.zip rewards.[0..adv-2] isDones.[0..adv-2]
                |> fun ls -> 
                    List.scanBack
                        (fun (reward,mask) adv -> reward + discount*adv*mask) 
                        ls 
                        (actorCritic.Value (states|> List.last) |> dsharp.view[agents.Length;1]) 

            actorCritic.Update(
                dsharp.stack states.[0..adv-1] |> dsharp.view [-1;4],
                dsharp.stack actions |> dsharp.view [-1;2], 
                dsharp.stack actionValues,
                agents=agents.Length,
                advanced=adv
            )
             

        
        //[
        //    for a in agents -> 
        //        Chart.Line(a.logSteps,Color=Color.fromString "purple",Opacity=0.2)  
        //] 
        //|> Chart.combine 
        Chart.Line(agents[0].logSteps,Color=Color.fromString "purple") // ,Opacity=0.2)  
        |> Chart.withSize(800.0,600.0)
        |> Chart.withLegend false
        |> Chart.withLayoutGridStyle(YGap= 0.1)
        |> Chart.withXAxisStyle "episode"  
        |> Chart.show 

        actorCritic 