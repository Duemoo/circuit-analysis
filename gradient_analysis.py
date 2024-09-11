import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from tqdm import tqdm
import plotly.io as pio
import re
import argparse

def load_checkpoint(checkpoint_path, step):
    path = os.path.join(checkpoint_path, f"checkpoint-{step}")
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)

def is_bias_or_layernorm(param_name):
    return 'bias' in param_name.lower() or 'ln' in param_name.lower()

@torch.no_grad()
def precompute_differences(checkpoint_path, start_step=0, end_step=999, omit_bias_layernorm=True, interval=1):
    print("Precomputing differences...")
    all_differences = []
    param_names = None
    param_shapes = None

    for step in tqdm(range(start_step, end_step)):
        if step%interval != 0:
            continue
        model1 = load_checkpoint(checkpoint_path, step)
        model2 = load_checkpoint(checkpoint_path, step + 1)

        step_differences = {}
        
        if param_names is None:
            param_names = [name for name, _ in model1.named_parameters() 
                           if not (omit_bias_layernorm and is_bias_or_layernorm(name))]
            param_shapes = {name: param.shape for name, param in model1.named_parameters() 
                            if not (omit_bias_layernorm and is_bias_or_layernorm(name))}

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2, f"Parameter names do not match: {name1} vs {name2}"
            
            if omit_bias_layernorm and is_bias_or_layernorm(name1):
                continue
            
            diff = param2.data - param1.data
            diff_np = diff.cpu().numpy()
            step_differences[name1] = diff_np

        all_differences.append(step_differences)
        
        del model1, model2
        torch.cuda.empty_cache()

    return all_differences, param_names, param_shapes

@torch.no_grad()
def precompute_parameter_values(checkpoint_path, start_step=0, end_step=999, omit_bias_layernorm=False, interval=1):
    print("Precomputing parameter values...")
    all_parameter_values = []
    param_names = None
    param_shapes = None

    for step in tqdm(range(start_step, end_step + 1)):
        if step%interval != 0:
            continue
        model = load_checkpoint(checkpoint_path, step)

        step_values = {}
        
        if param_names is None:
            param_names = [name for name, _ in model.named_parameters() 
                           if not (omit_bias_layernorm and is_bias_or_layernorm(name))]
            param_shapes = {name: param.shape for name, param in model.named_parameters() 
                            if not (omit_bias_layernorm and is_bias_or_layernorm(name))}

        for name, param in model.named_parameters():
            if omit_bias_layernorm and is_bias_or_layernorm(name):
                continue
            
            param_np = param.cpu().numpy()
            step_values[name] = param_np

        all_parameter_values.append(step_values)
        
        del model
        torch.cuda.empty_cache()

    return all_parameter_values, param_names, param_shapes


def get_histogram_ranges(all_differences, param_names):
    all_diffs = np.concatenate([diff[name].flatten() for diff in all_differences for name in param_names])
    x_min, x_max = np.percentile(all_diffs, [0.1, 99.9])
    y_max = 0
    
    for diff in all_differences:
        all_diffs_flat = np.concatenate([diff[name].flatten() for name in param_names])
        hist, _ = np.histogram(all_diffs_flat, bins=50, range=(x_min, x_max))
        y_max = max(y_max, hist.max())
    
    return x_min, x_max, y_max

def create_interactive_plot_diffs(all_differences, param_names, param_shapes, interval, step):
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols + 1  # +1 for the histogram

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=param_names + ["Distribution of All Parameter Differences"])

    all_diffs_flat = np.concatenate([all_differences[0][name].flatten() for name in param_names])
    vmin, vmax = np.percentile(all_diffs_flat, [1, 99])

    for idx, name in enumerate(param_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        diff = all_differences[0][name]
        
        if len(param_shapes[name]) == 1:  # For 1D tensors
            heatmap = go.Heatmap(z=[diff], colorscale='RdBu', zmin=vmin, zmax=vmax, zmid=0)
        else:
            heatmap = go.Heatmap(z=diff, colorscale='RdBu', zmin=vmin, zmax=vmax, zmid=0)
        
        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(title_text=f"Shape: {param_shapes[name]}", row=row, col=col)
        fig.update_yaxes(title_text=name, row=row, col=col)

    # Get fixed ranges for histogram
    x_min, x_max, y_max = get_histogram_ranges(all_differences, param_names)

    # Plot histogram of all differences
    fig.add_trace(go.Histogram(x=all_diffs_flat, nbinsx=50, autobinx=False, xbins=dict(start=x_min, end=x_max, size=(x_max-x_min)/50)), row=n_rows, col=1)
    fig.update_xaxes(title_text="Difference Value", range=[x_min, x_max], row=n_rows, col=1)
    fig.update_yaxes(title_text="Frequency", range=[0, y_max * 1.1], row=n_rows, col=1)

    fig.update_layout(height=300*n_rows, width=1200)

    # Add frames for animation
    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=[diff_dict[name]] if len(param_shapes[name]) == 1 else diff_dict[name])
                for name in param_names
            ] + [
                go.Histogram(x=np.concatenate([diff_dict[name].flatten() for name in param_names]))
            ],
            name=f'frame_{i}'
        )
        for i, diff_dict in enumerate(all_differences)
    ]
    fig.frames = frames

    # Add slider and play button
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": interval, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f'frame_{i}'],
                        {"frame": {"duration": interval, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": interval}}
                    ],
                    "label": str(i),
                    "method": "animate"
                }
                for i in range(len(all_differences))
            ]
        }
    ]

    # Add play and pause buttons
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": interval,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders
    )

    return fig


def create_interactive_plot_params(all_parameter_values, param_names, param_shapes, interval, step):
    n_params = len(param_names)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols + 1  # +1 for the histogram

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=param_names + ["Distribution of All Parameter Values"])

    all_values_flat = np.concatenate([all_parameter_values[0][name].flatten() for name in param_names])
    vmin, vmax = np.percentile(all_values_flat, [1, 99])

    for idx, name in enumerate(param_names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        values = all_parameter_values[0][name]
        
        if len(param_shapes[name]) == 1:  # For 1D tensors
            heatmap = go.Heatmap(z=[values], colorscale='RdBu', zmin=vmin, zmax=vmax, zmid=0)
        else:
            heatmap = go.Heatmap(z=values, colorscale='RdBu', zmin=vmin, zmax=vmax, zmid=0)
        
        fig.add_trace(heatmap, row=row, col=col)
        fig.update_xaxes(title_text=f"Shape: {param_shapes[name]}", row=row, col=col)
        fig.update_yaxes(title_text=name, row=row, col=col)

    # Get fixed ranges for histogram
    x_min, x_max, y_max = get_histogram_ranges(all_parameter_values, param_names)

    # Plot histogram of all parameter values
    fig.add_trace(go.Histogram(x=all_values_flat, nbinsx=50, autobinx=False, xbins=dict(start=x_min, end=x_max, size=(x_max-x_min)/50)), row=n_rows, col=1)
    fig.update_xaxes(title_text="Parameter Value", range=[x_min, x_max], row=n_rows, col=1)
    fig.update_yaxes(title_text="Frequency", range=[0, y_max * 1.1], row=n_rows, col=1)

    fig.update_layout(height=300*n_rows, width=1200)

    # Add frames for animation
    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=[value_dict[name]] if len(param_shapes[name]) == 1 else value_dict[name])
                for name in param_names
            ] + [
                go.Histogram(x=np.concatenate([value_dict[name].flatten() for name in param_names]),
                             nbinsx=50, autobinx=False, xbins=dict(start=x_min, end=x_max, size=(x_max-x_min)/50))
            ],
            name=f'frame_{i}'
        )
        for i, value_dict in enumerate(all_parameter_values)
    ]
    fig.frames = frames

    # Add slider and play button
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": interval, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f'frame_{i}'],
                        {"frame": {"duration": interval, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": interval}}
                    ],
                    "label": str(i),
                    "method": "animate"
                }
                for i in range(len(all_parameter_values))
            ]
        }
    ]

    # Add play and pause buttons
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": interval, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": interval,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        title_text="Parameter Values Animation"
    )

    return fig


def compute_activations(model, input_str, tokenizer):
    inputs = tokenizer(input_str, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions
    
    num_layers = len(hidden_states)
    num_attention_layers = len(attentions)
    num_heads = attentions[0].shape[1] if attentions else 1
    
    # print(f"Number of attention heads: {num_heads}")
    # print(f"Total number of layers (including input embedding): {num_layers}")
    # print(f"Number of attention layers: {num_attention_layers}")

    activations = []
    
    for layer_idx in range(num_layers):
        layer_activations = {
            'mlp': hidden_states[layer_idx][0].cpu().numpy(),
            'attention': []
        }
        
        # Add attention maps for layers that have them
        if layer_idx < num_attention_layers:
            layer_activations['attention'] = [attentions[layer_idx][0, head].cpu().numpy() for head in range(num_heads)]
            
        activations.append(layer_activations)
    
    return activations, input_tokens

def compute_activations_sequential(checkpoint_path, start_step, end_step, interval, input_str):
    all_activations = []
    input_tokens = None
    
    steps_to_compute = range(start_step, end_step + 1, interval)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(checkpoint_path, f"checkpoint-{start_step}"))
    
    for step in tqdm(steps_to_compute, desc="Computing activations"):
        model_path = os.path.join(checkpoint_path, f"checkpoint-{step}")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        
        activations, tokens = compute_activations(model, input_str, tokenizer)
        all_activations.append(activations)
        
        if input_tokens is None:
            input_tokens = tokens
        
        del model
        torch.cuda.empty_cache()
    
    return all_activations, input_tokens


def create_animated_activation_maps(all_activations_list, input_tokens_list, interval=200):
    num_inputs = len(all_activations_list)
    num_steps = len(all_activations_list[0])
    num_layers = len(all_activations_list[0][0])
    
    # Count total number of subplots (MLP + attention for each layer)
    total_subplots = sum(1 + len(layer['attention']) for layer in all_activations_list[0][0])
    
    # Create subplot titles
    subplot_titles = []
    for i in range(num_layers):
        subplot_titles.extend([f"MLP {i}"] + [f"Attention {i}"] * len(all_activations_list[0][0][i]['attention']))
    
    fig = make_subplots(
        rows=num_inputs, 
        cols=total_subplots,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.07,
        column_widths=[1] * total_subplots
    )
    
    # Adjust subplot titles to appear only above the first row
    for i, annotation in enumerate(fig['layout']['annotations']):
        if i < total_subplots:  # Only adjust the first row of titles
            annotation['y'] = 1.0  # Move the title above the plot
            annotation['yanchor'] = 'bottom'
        else:
            # Remove titles for other rows
            annotation['text'] = ''
    
    # Calculate column-wise min and max values for consistent scaling
    column_min_max = []
    for col in range(total_subplots):
        col_min = float('inf')
        col_max = float('-inf')
        for input_idx in range(num_inputs):
            for step in range(num_steps):
                layer = col // 2
                if col % 2 == 0:  # MLP column
                    data = all_activations_list[input_idx][step][layer]['mlp']
                else:  # Attention column
                    att_idx = (col - 1) % len(all_activations_list[0][0][layer]['attention'])
                    data = all_activations_list[input_idx][step][layer]['attention'][att_idx]
                col_min = min(col_min, data.min())
                col_max = max(col_max, data.max())
        column_min_max.append((col_min, col_max))
    
    # Function to create heatmap trace with column-wise consistent scaling
    def create_heatmap(z, col, showscale=False):
        return go.Heatmap(z=z, colorscale='Viridis', showscale=showscale, 
                          zmin=column_min_max[col][0], zmax=column_min_max[col][1])
    
    # Add initial data
    for input_idx in range(num_inputs):
        col = 0
        for layer in range(num_layers):
            # Add MLP activation
            fig.add_trace(create_heatmap(all_activations_list[input_idx][0][layer]['mlp'], col), 
                          row=input_idx+1, col=col+1)
            col += 1
            
            # Add attention heads if they exist for this layer
            for att in all_activations_list[input_idx][0][layer]['attention']:
                fig.add_trace(create_heatmap(att, col), row=input_idx+1, col=col+1)
                col += 1
    
    # Create frames for animation
    frames = []
    for step in range(num_steps):
        frame_data = []
        for input_idx in range(num_inputs):
            col = 0
            for layer in range(num_layers):
                # MLP activations
                frame_data.append(create_heatmap(all_activations_list[input_idx][step][layer]['mlp'], col))
                col += 1
                
                # Attention head activations
                for att in all_activations_list[input_idx][step][layer]['attention']:
                    frame_data.append(create_heatmap(att, col))
                    col += 1
        
        frames.append(go.Frame(data=frame_data, name=f'step_{step}'))
    
    # Update layout
    fig.update_layout(
        height=300 * num_inputs,
        width=250 * total_subplots,
        title_text="Activation and Attention Maps Animation",
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': interval, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate',
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate',
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Step:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f'step_{i}'], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(num_steps)
            ]
        }]
    )
    
    # Update axes and add input sequence annotations
    for input_idx in range(num_inputs):
        input_text = ' '.join(input_tokens_list[input_idx])
        fig.add_annotation(
            text=f"Input {input_idx + 1}: {input_text}",
            xref="paper", yref="paper",
            x=-0.06, y=1 - (input_idx + 0.5) / num_inputs,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(size=12),
            textangle=-90
        )
        for i in range(1, total_subplots + 1):
            fig.update_xaxes(
                title_text="Hidden Dim" if i % 2 == 1 else "Key",
                title_standoff=5,  # Reduces space between axis and its label
                row=input_idx+1,
                col=i
            )
            fig.update_yaxes(
                title_text="Sequence" if i % 2 == 1 else "Query",
                title_standoff=5,  # Reduces space between axis and its label
                row=input_idx+1,
                col=i
            )
    
    # Add column-wise colorbars
    for col in range(total_subplots):
        fig.add_layout_image(
            dict(
                source="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
                x=1 + (col / total_subplots),
                sizex=1/total_subplots,
                y=1,
                sizey=1,
                xref="paper",
                yref="paper",
                opacity=1.0,
                layer="below",
                sizing="stretch",
            )
        )
        fig.add_annotation(
            x=1 + ((col + 0.5) / total_subplots),
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Min: {column_min_max[col][0]:.2f}<br>Max: {column_min_max[col][1]:.2f}",
            showarrow=False,
            font=dict(size=8),
        )
    
    fig.frames = frames
    
    return fig


def get_latest_checkpoint_number(folder_path):
    # Get all items in the directory
    items = os.listdir(folder_path)
    
    # Filter for checkpoint folders and extract numbers
    checkpoint_numbers = []
    for item in items:
        match = re.match(r'checkpoint-(\d+)', item)
        if match:
            checkpoint_numbers.append(int(match.group(1)))
    
    # Return the maximum number, or None if no checkpoints found
    return max(checkpoint_numbers) if checkpoint_numbers else None

def save_animated_figure(fig, filename):
    pio.write_html(fig, file=filename, auto_play=False)
    print(f"Figure saved to {filename}")
    

def get_latest_checkpoint_number(folder_path):
    # Get all items in the directory
    items = os.listdir(folder_path)
    
    # Filter for checkpoint folders and extract numbers
    checkpoint_numbers = []
    for item in items:
        match = re.match(r'checkpoint-(\d+)', item)
        if match:
            checkpoint_numbers.append(int(match.group(1)))
    
    # Return the maximum number, or None if no checkpoints found
    return max(checkpoint_numbers) if checkpoint_numbers else None    

    
def main(args):
    # Precompute differences
    if args.auto_find_step:
        end_step = get_latest_checkpoint_number(args.exp_name)
    else:
        end_step = args.end_step
        
    print(f"end step: {end_step}")
    all_differences, param_names, param_shapes = precompute_differences(args.exp_name, args.start_step, end_step, interval=args.interval)
    all_parameter_values, param_names, param_shapes = precompute_parameter_values(args.exp_name, args.start_step, end_step, omit_bias_layernorm=True, interval=args.interval)
    diff_fig = create_interactive_plot_diffs(all_differences, param_names, param_shapes, interval=1, step=10)
    param_fig = create_interactive_plot_params(all_parameter_values, param_names, param_shapes, interval=1, step=10)
    if len(args.exp_name.split('/')[-1] )>0:
        fname = args.exp_name.split('/')[-1] 
    else:
        fname = args.exp_name.split('/')[-2]
    save_animated_figure(diff_fig, os.path.join(args.save_dir, fname + '_diff.html'))
    save_animated_figure(param_fig, os.path.join(args.save_dir, fname + '_param.html'))
    
    input_list = [
        "0010101010",
        "0010111010",
        "00101a1010"
        ]
    all_activations_list = []
    input_tokens_list = []
    for input_str in input_list:
        all_activations, input_tokens = compute_activations_sequential(
            args.exp_name, start_step=args.start_step, end_step=end_step, interval=args.interval, input_str=input_str
        )
        all_activations_list.append(all_activations)
        input_tokens_list.append(input_tokens)

    attn_fig = create_animated_activation_maps(all_activations_list, input_tokens_list, interval=args.interval)
    save_animated_figure(attn_fig, os.path.join(args.save_dir, fname + '_attn.html'))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--start_step", type=int, default=0, help="Starting step for difference computation")
    parser.add_argument("--end_step", type=int, default=9999, help="Ending step for difference computation")
    parser.add_argument("--interval", type=int, default=1, help="Interval between steps")
    parser.add_argument("--save_dir", type=str, default='/mnt/sda/hoyeon/circuit-analysis/plotly_figs')
    parser.add_argument("--auto_find_step", action='store_true')
    args = parser.parse_args()
    
    main(args)