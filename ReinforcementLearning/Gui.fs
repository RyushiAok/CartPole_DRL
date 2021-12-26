namespace CartPole

open System 

open Avalonia
open Avalonia.Controls
open Avalonia.Controls.Shapes
open Avalonia.FuncUI.DSL 
open Avalonia.FuncUI.Elmish
open Elmish 
open Avalonia.FuncUI.Components.Hosts
open Avalonia.Themes.Fluent
open Avalonia.Controls.ApplicationLifetimes 
open Avalonia.Input

open FSharp.Control
open FSharp.Control.Reactive 

open DiffSharp

module Gui =  
    type Msg = 
    | Update  of Apparatus 

    let update (msg: Msg) (state: Apparatus) : Apparatus =
        match msg with  
        | Update s ->  s
 
    let view (state: Apparatus) (dispatch) = 
        let cx = state.width / 2.0
        let cy = state.height * 0.75
        Canvas.create [ 
            Canvas.width  state.width
            Canvas.height state.height
            Canvas.background "white" 
            Canvas.children ([ 
                let x = cx + state.scale * float state.x
                let y = cy
                let l = 2.0 * float state.length * state.scale 
                let th = state.theta   
                Rectangle.create [
                    Rectangle.fill "black"
                    Rectangle.width  50.0
                    Rectangle.height 30.0
                    Rectangle.left (x - 25.0)
                    Rectangle.top  (cy - 15.0)
                ]
                Line.create [
                    Line.startPoint (x, cy)
                    Line.endPoint (x + l * sin th, y - l * cos th ) 
                    Line.stroke "orange"
                    Line.strokeThickness 7.5
                ]
            ])
        ] 

    type MainWindow() as this =
        inherit HostWindow()
        do       

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
                            Threading.Thread.Sleep 20
                            let action = agent.SelectAction(dsharp.tensor <| [env.State()]).toInt32()
                            env.Reflect(action |> function | 0 -> Left | _ -> Right) |> ignore 
                } 
                |> Async.Start 
                subject 
           
            let subject =  
                a2c() 
                //duelingNetwork()
                
                //let steps = 200
                //let env = Environment(steps) 
                //let subject = env.Subject 
                //subject

            let draw (initialState:Apparatus) =  
                let sub dispatch =   
                    Observable.interval (TimeSpan.FromMilliseconds 5.0) 
                    |> Observable.withLatestFrom (fun _ b -> b) subject
                    |> Observable.distinctUntilChanged
                    |> Observable.subscribe (fun x -> Msg.Update x |> dispatch)
                    |> ignore
                Cmd.ofSub sub
 
            let keyDownHandler (initialState: Apparatus) =  
                //Observable.interval (TimeSpan.FromMilliseconds 15.0)
                //|> Observable.subscribe(fun _ -> 
                //    subject.OnNext (subject.Value.Move None ))
                //|> ignore
                let sub dispatch =  
                    this.KeyDown.Add (fun eventArgs ->   
                        match eventArgs.Key with
                        | Key.A -> subject.OnNext <| subject.Value.Move  Left  
                        | Key.D -> subject.OnNext <| subject.Value.Move  Right 
                        | _     -> subject.OnNext <| Apparatus.init () 
                    )
                    |> ignore 
                Cmd.ofSub sub 

            base.Width  <- subject.Value.width
            base.Height <- subject.Value.height 

            Program.mkSimple (fun () -> subject.Value) update view
            |> Program.withHost this 
            |> Program.withSubscription draw 
            |> Program.withSubscription keyDownHandler 
            |> Program.run


    type App() =
        inherit Application()

        override this.Initialize() =
            this.Styles.Add (FluentTheme(baseUri = null, Mode = FluentThemeMode.Dark))

        override this.OnFrameworkInitializationCompleted() =
            match this.ApplicationLifetime with
            | :? IClassicDesktopStyleApplicationLifetime as desktopLifetime ->
                let mainWindow = MainWindow()
                desktopLifetime.MainWindow <- mainWindow
            | _ -> ()


    let appRun  argv =  
        AppBuilder
            .Configure<App>()
            .UsePlatformDetect()
            .UseSkia()
            .StartWithClassicDesktopLifetime(argv)