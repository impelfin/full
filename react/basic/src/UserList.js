import React from 'react';

function User({ user }) {
  return (
    <div>
      <b>{user.username}</b> &nbsp;&nbsp;
      <span>{user.email}</span>
    </div>
  );
}

function UserList({ users }) {
  return (
    <div>
      {users.map(user => (
        <User user={user} key={user.id} />
      ))}
    </div>
  );
}

export default UserList;
