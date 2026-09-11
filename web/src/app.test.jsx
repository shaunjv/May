import React from 'react';
import {render, screen, fireEvent, waitFor} from '@testing-library/react';
import {describe, it, expect, vi} from 'vitest';
import {Composer, PlanCard, preferredWorkspace} from './main';

describe('single composer', () => {
  it('sends on Enter and clears only after success', async () => {
    const send = vi.fn().mockResolvedValue();
    render(<Composer onSend={send}/>);
    fireEvent.change(screen.getByRole('textbox'), {target:{value:'hello'}});
    fireEvent.keyDown(screen.getByRole('textbox'), {key:'Enter'});
    await waitFor(() => expect(send).toHaveBeenCalledWith('hello'));
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''));
  });
  it('preserves Shift+Enter and blocks duplicate/busy sends', () => {
    const send = vi.fn();
    render(<Composer onSend={send} disabled/>);
    fireEvent.change(screen.getByRole('textbox'), {target:{value:'hello'}});
    fireEvent.keyDown(screen.getByRole('textbox'), {key:'Enter',shiftKey:true});
    fireEvent.keyDown(screen.getByRole('textbox'), {key:'Enter'});
    expect(send).not.toHaveBeenCalled();
  });
  it('keeps the draft after a failed request', async () => {
    render(<Composer onSend={vi.fn().mockRejectedValue(new Error('offline'))}/>);
    fireEvent.change(screen.getByRole('textbox'), {target:{value:'keep me'}});
    fireEvent.click(screen.getByLabelText('Send message'));
    await waitFor(() => expect(screen.getByLabelText('Send message')).not.toBeDisabled());
    expect(screen.getByRole('textbox')).toHaveValue('keep me');
  });
  it('requests cancellation', () => {
    const cancel = vi.fn();
    render(<Composer onSend={vi.fn()} onCancel={cancel}/>);
    fireEvent.click(screen.getByLabelText('Stop request'));
    expect(cancel).toHaveBeenCalledOnce();
  });
});

describe('approval card', () => {
  const task = {id:'t',state:'AWAITING_APPROVAL',plan:{goal:'Create file',phases:[{steps:[{step_id:'s',description:'Write file'}]}]},package:{plan_version:1,actions:[{action_id:'a',tool_name:'file_write',tool_input:{path:'test.py',content:'print(42)'}}]},preconditions:{'test.py':null},review_hash:'abc'};
  it('shows exact file content and requests explicit approval', () => {
    const action = vi.fn();
    render(<PlanCard task={task} onAction={action}/>);
    expect(screen.getByText('print(42)')).toBeInTheDocument();
    expect(action).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText('Approve & run'));
    expect(action).toHaveBeenCalledWith(task,'approve');
  });
  it('never offers approval on a terminal task', () => {
    render(<PlanCard task={{...task,state:'COMPLETED'}} onAction={vi.fn()}/>);
    expect(screen.queryByText('Approve & run')).not.toBeInTheDocument();
  });
});

describe('workspace selection', () => {
  it('keeps the active conversation workspace instead of returning to the launch folder', () => {
    expect(preferredWorkspace({workspace: 'C:\\Users\\Shaun\\Downloads\\anime'}, 'E:\\shaun\\projects\\agent')).toBe('C:\\Users\\Shaun\\Downloads\\anime');
  });
  it('uses the launch folder only before a conversation exists', () => {
    expect(preferredWorkspace(null, 'E:\\shaun\\projects\\agent')).toBe('E:\\shaun\\projects\\agent');
  });
});
