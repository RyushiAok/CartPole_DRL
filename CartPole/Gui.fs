﻿namespace CartPole

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
open System.Reactive.Subjects

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

    type MainWindow(subject: BehaviorSubject<Apparatus>) as this =
        inherit HostWindow()
        do     
            let draw (initialState:Apparatus) =  
                let sub dispatch =   
                    Observable.interval (TimeSpan.FromMilliseconds 5.0) 
                    |> Observable.withLatestFrom (fun _ b -> b) subject
                    |> Observable.distinctUntilChanged
                    |> Observable.subscribe (fun x -> Msg.Update x |> dispatch)
                    |> ignore
                Cmd.ofSub sub
 
            let keyDownHandler (initialState: Apparatus) = 
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


    type App(subject) =
        inherit Application()

        override this.Initialize() =
            this.Styles.Add (FluentTheme(baseUri = null, Mode = FluentThemeMode.Dark))

        override this.OnFrameworkInitializationCompleted() =
            match this.ApplicationLifetime with
            | :? IClassicDesktopStyleApplicationLifetime as desktopLifetime ->
                let mainWindow = MainWindow(subject)
                desktopLifetime.MainWindow <- mainWindow
            | _ -> ()


    let appRun (subject: BehaviorSubject<Apparatus>) argv =   
        AppBuilder
            .Configure(fun () -> App subject)
            .UsePlatformDetect()
            .UseSkia()
            .StartWithClassicDesktopLifetime(argv)